/*************************************************************************
 * Copyright (c) 2016-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_PRIMITIVES_H_
#define NCCL_PRIMITIVES_H_

#include <type_traits>
#include "reduce_kernel.h" // for reduction funcs
#include "common.h"
#include <curand_kernel.h>

#define SPINS_BEFORE_CHECK_ABORT 1000000

// Unroll unconditionally the first send/recv since nsend/nrecv should be at
// least 1 if SEND/RECV is set.
#define FOR_SEND(func, ...) do { \
  if (SEND) { \
    /* Send to far first, then close */ \
    for (int i=1; i<NSEND && i<nsend; i++) func(i, ##__VA_ARGS__); \
    func(0, ##__VA_ARGS__); \
  } \
} while (0)

#define FOR_RECV(func, ...) do { \
  if (RECV) { \
    /* Recv from close first, then far */ \
    func(0, ##__VA_ARGS__); \
    for (int i=1; i<NRECV && i<nrecv; i++) func(i, ##__VA_ARGS__); \
  } \
} while (0)

// Implementation of primitive types
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, class FUNC>
class ncclPrimitives {
 private:
  const int tid;
  const int nthreads;
  const int wid;
  const int stepSize;
  int nrecv = 0;
  int nsend = 0;
  struct ncclConnInfo* recvConn = NULL;
  volatile uint64_t* recvConnHeadPtr = NULL;
  uint64_t recvConnHead;
  volatile uint64_t* recvConnTailPtr = NULL;
  uint64_t recvConnTail;
  uint64_t recvConnTailCache; // Cache last seen value
  volatile int* fineGrainedOverlapSyncMem;
  int previousOverlapValue;
  curandState* randState;
  int matrixCols;
  int matrixRows;

  struct ncclConnInfo* sendConn = NULL;
  volatile int* sendConnFifoPtr = NULL;
  volatile uint64_t* sendConnTailPtr = NULL;
  uint64_t sendConnTail;
  volatile uint64_t* sendConnHeadPtr = NULL;
  uint64_t sendConnHead;
  uint64_t sendConnHeadCache; // Cache last seen value

  uint64_t recvStep[NRECV];
  uint64_t sendStep[NSEND];
  const T* recvDirectBuff[NRECV];
  T* sendDirectBuff[NSEND];
  const T* recvBuff[NRECV];
  T* sendBuff[NSEND];
  struct ncclDevComm* comm;

  inline __device__ int recvOffset(int i) { return (recvStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ int sendOffset(int i) { return (sendStep[i]%NCCL_STEPS)*stepSize; }
  inline __device__ const T* recvPtr(int i) { return ((const T*)recvBuff[i])+recvOffset(i); }
  inline __device__ T* sendPtr(int i) { return ((T*)sendBuff[i])+sendOffset(i); }

  inline __device__ void barrier() {
    asm volatile ("bar.sync 1, %0;" :: "r"(nthreads));
  }
  inline __device__ void subBarrier() {
    asm volatile ("bar.sync 2, %0;" :: "r"(nthreads-WARP_SIZE));
  }

  uint32_t mismatch = 0;
  const uint64_t opCount;

  inline __device__ void checkMismatch(struct ncclConnInfo* conn) {
    if (mismatch) {
      // In non-LL, we use _threadfence_system before incrementing opCount, yet we are still waiting for credits here, so there must be a size mismatch
      *(comm->fatalDevError) = ncclDevAssertedMismatch;
    } else if (conn && *conn->opCountRem > opCount) {
      mismatch += 1;
    }
  }

  uint32_t spins = 0;
  uint32_t abort = 0;

  inline __device__ int checkAbort(int i, int send) {
    spins++;
    if (abort == 0 && spins == SPINS_BEFORE_CHECK_ABORT) {
      abort = *(comm->abortFlag);
      if (wid == i) checkMismatch(send ? sendConn : recvConn);
      spins = 0;
    }
    return abort;
  }

  inline __device__ void waitSend(int nbytes) {
    spins = 0;
    mismatch = 0;
    if (sendConnHeadPtr) {
      while (sendConnHeadCache + NCCL_STEPS < sendConnHead + SLICESTEPS) {
        sendConnHeadCache = *sendConnHeadPtr;
        if (checkAbort(wid, 1)) break;
      }
      if (sendConnFifoPtr) {
        sendConnFifoPtr[sendConnHead%NCCL_STEPS] = nbytes;
      }
      sendConnHead += SLICESTEPS;
    }
  }

  inline __device__ void waitRecv() {
    spins = 0;
    mismatch = 0;
    if (recvConnTailPtr) {
      while (recvConnTailCache < recvConnTail + SLICESTEPS) {
        recvConnTailCache = *recvConnTailPtr;
        if (checkAbort(wid, 0)) break;
      }
      recvConnTail += SLICESTEPS;
    }
  }

  inline __device__ void incRecv(int i) {
    recvStep[i] += SLICESTEPS;
  }
  inline __device__ void postRecv() {
    if (recvConnHeadPtr) *recvConnHeadPtr = recvConnHead += SLICESTEPS;
  }

  inline __device__ void incSend(int i) {
    sendStep[i] += SLICESTEPS;
  }
  inline __device__ void postSend() {
    if (sendConnTailPtr) *sendConnTailPtr = sendConnTail += SLICESTEPS;
  }

  template <int DIRECTRECV>
  inline __device__ const T* directRecvPtr(int i, ssize_t directOffset) {
    return DIRECTRECV && recvDirectBuff[i] ? recvDirectBuff[i]+directOffset : recvPtr(i);
  }

  template <int DIRECTSEND>
  inline __device__ T* directSendPtr(int i, ssize_t directOffset) {
    return DIRECTSEND && sendDirectBuff[i] ? sendDirectBuff[i]+directOffset : sendPtr(i);
  }

  template <int DIRECTRECV>
  inline __device__ int directRecvInc(int i, int directInc, int sliceInc) {
    return DIRECTRECV && recvDirectBuff[i] ? directInc : sliceInc;
  }

  template <int DIRECTSEND>
  inline __device__ int directSendInc(int i, int directInc, int sliceInc) {
    return DIRECTSEND && sendDirectBuff[i] ? directInc : sliceInc;
  }

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST, int WEIGHT_UPDATE, int MATMUL_OVERLAP, int DROPOUT_BIAS_LAYERNORM>
  inline __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, T* firstMoment, T* secondMoment, int nelem, ssize_t directOffset, const T alpha, 
            const T beta1, const T beta2, const int epoch, const int overlapExpectedValue) {
    //firstMoment is Bias when DROPOUT_BIAS_LAYERNORM is 1
    //secondMoment is addTensor when DROPOUT_BIAS_LAYERNORM is 1
    //epoch is biasSize when DROPOUT_BIAS_LAYERNORM is 1

    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

    const T* srcs[RECV*NRECV+SRC];
    srcs[0] = SRC ? srcPtr : directRecvPtr<DIRECTRECV>(0, directOffset);
    if (RECV) {
      if (SRC) srcs[1] = recvPtr(0);
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);
    }

    T* dsts[SEND*NSEND+DST];
    dsts[0] = DST ? dstPtr : directSendPtr<DIRECTSEND>(0, directOffset);
    if (SEND) {
      if (DST) dsts[1] = directSendPtr<DIRECTSEND>(0, directOffset);
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = directSendPtr<DIRECTSEND>(i, directOffset);
    }

    bool syncThread = tid >= nthreads-WARP_SIZE;

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (!syncThread) {
        if (SEND) waitSend(realSize*sizeof(T));
        if (RECV) waitRecv();
        if (realSize > 0) {
          if (slice == 0 && MATMUL_OVERLAP == 1 && threadIdx.x == 0) {
            if (previousOverlapValue < overlapExpectedValue) {
              while ((previousOverlapValue = *(fineGrainedOverlapSyncMem + blockIdx.x)) < overlapExpectedValue) {
                //Wait until rows to be used by this iteration has been generated by cuBLAS.
                // printf("value %d expected %d channel %d\n", previousOverlapValue, overlapExpectedValue, b);
              }
            }
          }
          subBarrier();
          if (DIRECTRECV && recvDirectBuff[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, NSEND, WEIGHT_UPDATE, 0,0, DROPOUT_BIAS_LAYERNORM>(tid, nthreads-WARP_SIZE, 1, srcs, nsend, dsts+1, firstMoment, secondMoment, nullptr, realSize, alpha, beta1, beta2, epoch, 0, 0, 0, nullptr,nullptr, 0, randState);
            }
          } else {
            ReduceOrCopyMulti<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST, WEIGHT_UPDATE, 0,0, DROPOUT_BIAS_LAYERNORM>(tid, nthreads-WARP_SIZE, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, firstMoment, secondMoment, nullptr, realSize, alpha, beta1, beta2, epoch, 0, 0, 0, nullptr,nullptr, 0, randState);
          }
        }
      }
      barrier();
      FOR_SEND(incSend);
      FOR_RECV(incRecv);
      if (syncThread) {
        if (SEND) {
          if (realSize > 0 && wid == 0) __threadfence_system();
          __syncwarp();
          postSend();
        }
        if (RECV) postRecv();
      }
      srcs[0] += SRC ? realSize : directRecvInc<DIRECTRECV>(0, realSize, sliceSize);
      for (int i=1-SRC; i<RECV*NRECV; i++) srcs[SRC+i] += sliceSize;
      dsts[0] += DST ? realSize : directSendInc<DIRECTSEND>(0, realSize, sliceSize);
      for (int i=1-DST; i<SEND*NSEND; i++) dsts[DST+i] += directSendInc<DIRECTSEND>(i, realSize, sliceSize);
      offset += realSize;
    }
  }


  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST, int WEIGHT_UPDATE, int MATMUL_OVERLAP, int DROPOUT_BIAS_LAYERNORM>
  inline __device__ void
  GenericOpMatrixBlock(const T* srcPtr, T* dstPtr, T* firstMoment, T* secondMoment, int nelem, ssize_t directOffset, 
                       int chunkStartRow, int chunkStartCol, int chunkRows, int chunkCols, int matrixM, int matrixN, const T alpha, 
                       const T beta1, const T beta2, const int epoch, const int overlapExpectedValue) {
    //firstMoment is Bias when DROPOUT_BIAS_LAYERNORM is 1
    //secondMoment is addTensor when DROPOUT_BIAS_LAYERNORM is 1
    //epoch is biasSize when DROPOUT_BIAS_LAYERNORM is 1

    int offset = 0;
    int sliceSize = stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);

    const T* srcs[RECV*NRECV+SRC];
    srcs[0] = SRC ? srcPtr : recvPtr(0);// directRecvPtr<DIRECTRECV>(0, directOffset);
    if (RECV) {
      if (SRC) srcs[1] = recvPtr(0);
      for (int i=1; i<NRECV && i<nrecv; i++) srcs[SRC+i] = recvPtr(i);
    }

    T* dsts[SEND*NSEND+DST];
    dsts[0] = DST ? dstPtr : sendPtr(0); //directSendPtr<DIRECTSEND>(0, directOffset);
    if (SEND) {
      if (DST) dsts[1] = sendPtr(0); //directSendPtr<DIRECTSEND>(0, directOffset);
      for (int i=1; i<NSEND && i<nsend; i++) dsts[DST+i] = sendPtr(i); //directSendPtr<DIRECTSEND>(i, directOffset);
    }

    bool syncThread = tid >= nthreads-WARP_SIZE;

    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));

      if (!syncThread) {
        if (SEND) waitSend(realSize*sizeof(T));
        if (RECV) waitRecv();
        if (realSize > 0) {
          if (slice == 0 && MATMUL_OVERLAP == 1 && threadIdx.x == 0) {
            if (previousOverlapValue < overlapExpectedValue) {
              while ((previousOverlapValue = *(fineGrainedOverlapSyncMem + blockIdx.x)) < overlapExpectedValue) {
                //Wait until rows to be used by this iteration has been generated by cuBLAS.
                // printf("value %d expected %d channel %d\n", previousOverlapValue, overlapExpectedValue, b);
              }
            }
          }
          subBarrier();
          if (DIRECTRECV && recvDirectBuff[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              ReduceOrCopyMultiMatrixBlock<UNROLL, FUNC, T, SRC, DST, 1, 1, 1, NSEND, WEIGHT_UPDATE, 0,0, DROPOUT_BIAS_LAYERNORM>(tid, nthreads-WARP_SIZE, 1, srcs, nsend, dsts+1, firstMoment, secondMoment, 
                nullptr, realSize, alpha, beta1, beta2, epoch, directOffset+offset, chunkStartRow, chunkStartCol, chunkRows, chunkCols, matrixM, matrixN, nullptr,nullptr, 0, randState);
            }
          } else {
            ReduceOrCopyMultiMatrixBlock<UNROLL, FUNC, T, SRC, DST, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST, WEIGHT_UPDATE, 0,0, DROPOUT_BIAS_LAYERNORM>(tid, nthreads-WARP_SIZE, RECV*nrecv+SRC, srcs, SEND*nsend+DST, dsts, firstMoment, secondMoment, 
                nullptr, realSize, alpha, beta1, beta2, epoch, directOffset+offset, chunkStartRow, chunkStartCol, chunkRows, chunkCols, matrixM, matrixN, nullptr,nullptr, 0, randState);
          }
        }
      }
      barrier();
      FOR_SEND(incSend);
      FOR_RECV(incRecv);
      if (syncThread) {
        if (SEND) {
          if (realSize > 0 && wid == 0) __threadfence_system();
          __syncwarp();
          postSend();
        }
        if (RECV) postRecv();
      }
      if (!SRC)
        srcs[0] += directRecvInc<DIRECTRECV>(0, realSize, sliceSize);
      for (int i=1-SRC; i<RECV*NRECV; i++) srcs[SRC+i] += sliceSize;
      if (!DST) 
        dsts[0] += directSendInc<DIRECTSEND>(0, realSize, sliceSize);
      for (int i=1-DST; i<SEND*NSEND; i++) dsts[DST+i] += directSendInc<DIRECTSEND>(i, realSize, sliceSize);
      offset += realSize;
    }
  }

  __device__ __forceinline__ void loadRecvConn(struct ncclConnInfo* conn, int i, T* directBuff) {
    recvBuff[i] = (const T*)conn->buff;
    recvStep[i] = conn->step;
    recvStep[i] = ROUNDUP(recvStep[i], SLICESPERCHUNK*SLICESTEPS);
    recvDirectBuff[i] = NULL;
    if (directBuff && (conn->direct & NCCL_DIRECT_GPU)) {
      recvDirectBuff[i] = directBuff;
      if (tid == 0) *conn->ptrExchange = directBuff;
    }
    if (wid == i) recvConn = conn;
    if (wid == i) recvConnTail = recvConnHead = recvStep[i]; // Make sure we set this after rounding up
    nrecv++;
  }
  __device__ __forceinline__ void loadRecvSync() {
    if (tid >= WARP_SIZE && tid < 2*WARP_SIZE && wid<nrecv) {
      recvConnTailPtr = recvConn->tail;
      recvConnTailCache = *recvConnTailPtr;
    }
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      recvConnHeadPtr = recvConn->head;
      // Return credits in case we rounded up.
      *recvConnHeadPtr = recvConnHead;
      // Update opCount in case we skipped some operations
      *(recvConn->opCountLoc) = opCount;
    }
  }

  __device__ __forceinline__ void loadSendConn(struct ncclConnInfo* conn, int i, T* directBuff) {
    sendBuff[i] = (T*)conn->buff;
    sendStep[i] = conn->step;
    sendStep[i] = ROUNDUP(sendStep[i], SLICESPERCHUNK*SLICESTEPS);
    sendDirectBuff[i] = NULL;
    if (directBuff && (conn->direct & NCCL_DIRECT_GPU)) {
      void* volatile* ptr = conn->ptrExchange;
      while ((sendDirectBuff[i] = (T*)(*ptr)) == NULL);
      barrier();
      if (tid == 0) *ptr = NULL;
    }
    if (wid == i) sendConn = conn;
    if (wid == i) sendConnTail = sendConnHead = sendStep[i]; // Make sure we set this after rounding up
    nsend++;
  }
  __device__ __forceinline__ void loadSendSync() {
    if (tid < nsend) {
      sendConnHeadPtr = sendConn->head;
      sendConnHeadCache = *sendConnHeadPtr;
      sendConnFifoPtr = sendConn->fifo;
      *(sendConn->opCountLoc) = opCount;
    }
    if (tid >= nthreads-WARP_SIZE && wid<nsend) {
      sendConnTailPtr = sendConn->tail;
    }
  }

  __device__ __forceinline__ void saveRecvSync() {
    if (tid >= nthreads-WARP_SIZE && wid < nrecv) {
      recvConn->step = recvConnHead;
      *(recvConn->opCountLoc) = opCount+1;
      __threadfence_system();
    }
  }

  __device__ __forceinline__ void saveSendSync() {
    if (tid < nsend) {
      sendConn->step = sendConnHead;
      *(sendConn->opCountLoc) = opCount+1;
      __threadfence_system();
    }
  }

 public:
  __device__ __forceinline__
  ncclPrimitives(const int tid, const int nthreads, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, const uint64_t opCount,
  volatile int* fineGrainedOverlapSyncMem, int matrixRows, int matrixCols)
    : comm(comm), tid(tid), nthreads(nthreads), wid(tid%WARP_SIZE), stepSize(stepSize), opCount(opCount), fineGrainedOverlapSyncMem(fineGrainedOverlapSyncMem), randState(nullptr),
    matrixCols(matrixCols), matrixRows(matrixRows) {
    previousOverlapValue = -1;
    // Make sure step is updated before we read it.
    barrier();
    for (int i=0; i<NRECV && recvPeers[i] >= 0; i++) loadRecvConn(&channel->devPeers[recvPeers[i]].recv.conn, i, directBuff);
    for (int i=0; i<NSEND && sendPeers[i] >= 0; i++) loadSendConn(&channel->devPeers[sendPeers[i]].send.conn, i, directBuff);
    loadRecvSync();
    loadSendSync();
  }

  __device__ __forceinline__ void setCurandState(curandState* _randState) {randState = _randState;}
  __device__ __forceinline__ void
  send(const T* src, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0, 0, 0, 0>(src, NULL, NULL, NULL, nelem, 0, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }

  __device__ __forceinline__ void
  directSend(const T* src, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 0, 0, 0, 0>(src, NULL, NULL, NULL, nelem, directOffset, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }

  __device__ __forceinline__ void
  recv(T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1, 0, 0, 0>(NULL, dst, NULL, NULL, nelem, 0, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }
  __device__ __forceinline__ void
  directRecv(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 0, 1, 0, 0, 1, 0, 0, 0>(NULL, dst, NULL, NULL, nelem, directOffset, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }

  __device__ __forceinline__ void
  copySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 1, 0, 0, 0>(src, dst, NULL, NULL, nelem, 0, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }
  __device__ __forceinline__ void
  directCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    GenericOp<0, 1, 0, 1, 1, 1, 0, 0, 0>(src, dst, NULL, NULL, nelem, directOffset, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }

  __device__ __forceinline__ void
  recvCopySend(T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1, 0, 0, 0>(NULL, dst, NULL, NULL, nelem, 0, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }
  __device__ __forceinline__ void
  directRecvCopySend(T* dst, ssize_t directOffset, int nelem) {
    GenericOp<1, 1, 1, 1, 0, 1, 0, 0, 0>(NULL, dst, NULL, NULL, nelem, directOffset, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }

  __device__ __forceinline__ void
  recvReduceCopy(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1, 0, 0, 0>(src, dst, NULL, NULL, nelem, 0, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }

  __device__ __forceinline__ void
  recvReduceSend(const T* src, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 0, 0, 0, 0>(src, NULL, NULL, NULL, nelem, 0, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }

  __device__ __forceinline__ void
  recvReduceSendOverlappedWithMatmul(const T* src, int nelem, int expectedVal) {
    GenericOp<0, 0, 1, 1, 1, 0, 0, 1, 0>(src, NULL, NULL, NULL, nelem, 0, (T)0.0, (T)0.0, (T)0.0, (T)0.0, expectedVal);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const T* src, T* dst, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 1, 0, 0, 0>(src, dst, NULL, NULL, nelem, 0, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }
  __device__ __forceinline__ void
  directRecvReduceCopySend(const T* src, T* dst, ssize_t directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1, 0, 0, 0>(src, dst, NULL, NULL, nelem, directOffset, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }

  __device__ __forceinline__ void 
  directRecvReduceCopySendAdam(const T* src, T* weight, T* firstMoment, T*secondMoment, 
                               ssize_t directOffset, int nelem, const T alpha, const T beta1, const T beta2, const int epoch) {
    GenericOp<0, 1, 1, 1, 1, 1, 1, 0, 0>(src, weight, firstMoment, secondMoment, nelem, directOffset, alpha, beta1, beta2, epoch, 0);
  }

  __device__ __forceinline__ void 
  directRecvReduceCopySendWeight(const T* src, T* weight, ssize_t directOffset, int nelem, const T alpha) {
    GenericOp<0, 1, 1, 1, 1, 1, 1, 0, 0>(src, weight, NULL, NULL, nelem, directOffset, alpha, (T)0.0, (T)0.0, (T)0.0, 0);
  }

  __device__ __forceinline__ void
  directRecvReduceCopySendDropoutBiasLayernorm(const T* src, T* dst, const T* addTensor, const T* bias, int biasSize, ssize_t directOffset, int nelem) {
    // Direct is only for the send part
    GenericOp<0, 1, 1, 1, 1, 1, 0, 0, 1>(src, dst, (T*)addTensor, (T*)bias, nelem, directOffset, (T)0.0, (T)0.0, (T)0.0, biasSize, 0);
  }

  /*
   * These functions work on a block of a matrix.
   * In the function call src and dst should not be incremented by offset.
   */
  __device__ __forceinline__ void
  sendMatrixBlock(const T* src, ssize_t offset, int chunkStartRow, int chunkStartCol, int chunkRows, int chunkCols, int matrixM, int matrixN, int nelem) {
    GenericOpMatrixBlock<0, 0, 0, 1, 1, 0, 0, 0, 0>(src, NULL, NULL, NULL, nelem, offset, chunkStartRow, chunkStartCol, chunkRows, chunkCols, matrixM, matrixN, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }

  __device__ __forceinline__ void
  recvMatrixBlock(T* dst, ssize_t offset, int chunkStartRow, int chunkStartCol, int chunkRows, int chunkCols, int matrixM, int matrixN, int nelem) {
    GenericOpMatrixBlock<0, 0, 1, 0, 0, 1, 0, 0, 0>(NULL, dst, NULL, NULL, nelem, offset, chunkStartRow, chunkStartCol, chunkRows, chunkCols, matrixM, matrixN, 
                                                    (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }

  __device__ __forceinline__ void
  recvCopySendMatrixBlock(T* dst, ssize_t offset, int chunkStartRow, int chunkStartCol, int chunkRows, int chunkCols, int matrixM, int matrixN, int nelem) {
    GenericOpMatrixBlock<0, 0, 1, 1, 0, 1, 0, 0, 0>(NULL, dst, NULL, NULL, nelem, offset, chunkStartRow, chunkStartCol, chunkRows, chunkCols, matrixM, matrixN,
                                                    (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }

  __device__ __forceinline__ void
  recvReduceSendMatrixBlockOverlappedWithMatmul(const T* src, ssize_t offset, int chunkStartRow, int chunkStartCol, int chunkRows, int chunkCols, int matrixM, int matrixN, int expectedVal, int nelem) {
    assert(false);
    // GenericOpMatrixBlock<0, 0, 1, 1, 1, 0, 0, 1, 0>(src, NULL, NULL, NULL, nelem, offset, 
    // chunkStartRow, chunkStartCol, chunkRows, chunkCols, matrixM, matrixN, (T)0.0, (T)0.0, (T)0.0, (T)0.0, expectedVal);
  }
  
  __device__ __forceinline__ void
  recvReduceSendMatrixBlock(const T* src, ssize_t offset, int chunkStartRow, int chunkStartCol, int chunkRows, int chunkCols, int matrixM, int matrixN,  int nelem) {
    GenericOpMatrixBlock<0, 0, 1, 1, 1, 0, 0, 0, 0>(src, NULL, NULL, NULL, nelem, offset, chunkStartRow, chunkStartCol, chunkRows, chunkCols, matrixM, matrixN, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }

    __device__ __forceinline__ void
  recvReduceCopySendMatrixBlock(const T* src, T* dst, ssize_t offset, int chunkStartRow, int chunkStartCol, int chunkRows, int chunkCols, int matrixM, int matrixN, int nelem) {
    // Direct is only for the send part
    GenericOpMatrixBlock<0, 0, 1, 1, 1, 1, 0, 0, 0>(src, dst, NULL, NULL, nelem, offset, 
                                         chunkStartRow, chunkStartCol, chunkRows, chunkCols, matrixM, matrixN, (T)0.0, (T)0.0, (T)0.0, (T)0.0, 0);
  }


  __device__ __forceinline__ ~ncclPrimitives() {
    // Save steps for the next operation
    saveRecvSync();
    saveSendSync();
  }
};

#include "scattered_primitives.h"
#include "prims_ll.h"
#include "scattered_prims_ll.h"
//#include "prims_ll128.h"

#endif
