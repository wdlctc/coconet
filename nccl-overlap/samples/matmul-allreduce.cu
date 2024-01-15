#include "header.h"
#include "cutlass-matmul.h"
#include <cuda_profiler_api.h>

#include <map>

#define MAX_CHANNELS 80

void pipe_rowmajorABC(cublasHandle_t handle, const half *alpha, const half *beta, const half* m1, const half* m2, half* m1m2, ncclComm_t comm, cudaStream_t stream, int M, int N, int K, float& allReduceTime, float& cublasTime) {
    cudaEvent_t startpipe, stoppipe;
    float elapsedTime = 0;
    // MPI_Barrier(MPI_COMM_WORLD);

    CUDACHECK(cudaEventCreate(&startpipe));
    CUDACHECK(cudaEventCreate(&stoppipe));
    CUDACHECK(cudaEventRecord(startpipe, stream));
  CUBLASCHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
    N, M, K, 
    alpha,
    m2, CUDA_R_16F, N,
    m1, CUDA_R_16F, K,
    beta, 
    m1m2, CUDA_R_16F, N,
    CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
    CUDACHECK(cudaEventRecord(stoppipe, stream));

    CUDACHECK(cudaStreamSynchronize(stream));
    CUDACHECK(cudaEventSynchronize(stoppipe));
    CUDACHECK(cudaEventElapsedTime(&elapsedTime, startpipe,stoppipe));

    cublasTime += elapsedTime;

  elapsedTime = 0;
  double t1 = getCurrentTime();

  NCCLCHECK(ncclAllReduceMatrix(m1m2, M*N, M, N, N, ncclHalf, ncclSum, comm, stream));
  CUDACHECK(cudaStreamSynchronize(stream));
  double t2 = getCurrentTime();
  allReduceTime += (t2-t1)*1000.0f;
}


bool mpiRef(const float* m1, const float* m2, float* m1m2, int M, int N, int K, int comm_size, int rank = -1)
{
  return true;
}


template<typename T>
std::vector<std::vector<std::tuple<int, int, int, int>>> getChunkBlocks
  (int rank, size_t matrixSize, int nranks, int* rings, int MATMUL_M, int MATMUL_N,
  const int realChunkCols, int& maxRealChunkRows) 
{
  std::vector<std::vector<std::tuple<int, int, int, int>>> chunkBlocks;

  assert (atoi(getenv ("NCCL_MIN_NCHANNELS")) == atoi(getenv ("NCCL_MAX_NCHANNELS")));
  int nChannels = atoi(getenv ("NCCL_MIN_NCHANNELS"));
  int nThreads = atoi(getenv("NCCL_NTHREADS"));
  // int nThreadsLL128 = atoi(getenv ("NCCL_LL128_NTHREADS"));
  int channelBuffSize = atoi(getenv("NCCL_BUFFSIZE"));

  const int stepSize = channelBuffSize / (sizeof(T)*NCCL_STEPS);
  const size_t chunkSize = stepSize * ALLREDUCE_CHUNKSTEPS;
  const ssize_t loopSize = nChannels*(ssize_t)chunkSize;
  maxRealChunkRows = 0;

  printf("matrixSize %d chunkSize %d nranks * loopSize %d\n", matrixSize, chunkSize, nranks * loopSize);
  for (int userRank = nranks - 1; userRank >= 0; userRank--) {
    chunkBlocks.push_back(std::vector<std::tuple<int, int, int, int>>());
    int combinedRanks = 1;
    for (int channel = 0; channel < nChannels; channel++) {
      //TODO: following loop only run for once right now.

      for (size_t gridOffset = 0; gridOffset < matrixSize; gridOffset += nranks * loopSize) {
        size_t realChunkSize = min(chunkSize, DIVUP(matrixSize-gridOffset,nranks*nChannels));
        if (matrixSize %3 == 0 && MATMUL_N != 12288) {
          ALIGN_SIZE(realChunkSize, nThreads*sizeof(uint64_t)/sizeof(T) * 3);
        } else 
        if (matrixSize % 12288 == 0) {
          ALIGN_SIZE(realChunkSize, nThreads*sizeof(uint64_t)/sizeof(T) * 12);
        }
        else {
          ALIGN_SIZE(realChunkSize, nThreads*sizeof(uint64_t)/sizeof(T));
        }

        const int realChunkRows = realChunkSize/realChunkCols;
        const int gridOffsetStartRow = gridOffset / MATMUL_N;

        maxRealChunkRows = std::max (maxRealChunkRows, realChunkRows);

        int chunkIdx = rings[channel*nranks + userRank] * nChannels + channel;
        int chunkStartRow = gridOffsetStartRow + chunkIdx / (MATMUL_N / realChunkCols) * realChunkRows;
        int chunkStartCol = chunkIdx % (MATMUL_N / realChunkCols) * realChunkCols;

        int nelem = min(realChunkSize, (matrixSize - (chunkStartRow * MATMUL_N + (MATMUL_M - chunkStartRow) * (MATMUL_N - (MATMUL_N - chunkStartCol)))));
        int chunkRows = min(min(nelem/realChunkCols, realChunkRows), MATMUL_M - chunkStartRow);
        int chunkCols;
        chunkCols = realChunkCols;
        nelem = chunkCols * chunkRows;

        chunkBlocks[chunkBlocks.size() - 1].push_back(std::make_tuple(chunkStartRow, chunkStartCol, chunkRows, chunkCols));
      }
    }
  }

  return chunkBlocks;
}

int main(int argc, char** argv){
  const int N_GPUS = 2;
  
  MPI_Init(&argc, &argv);
  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ncclComm_t comms;
  CUDACHECK(cudaSetDevice(rank));

  // initialize nccl
  ncclUniqueId id;
  if (rank == 0) {
    ncclGetUniqueId(&id);
  }
  MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comms, comm_size, id, rank);
  int ringLength;
  int nChannels;
  int* rings = new int[comm_size * MAX_CHANNELS];
  getNCCLRing(&comms, rings, ringLength, nChannels);
  // ncclGetRingCount(comms, &ringLength);

  for (int _rank = 0; _rank < comm_size; _rank++) {
    if (_rank != rank) continue;
    std::cout << "rank: " << rank << ":";
    for (int i = 0; i < ringLength; i++) {
      std::cout << rings[i] << "->";
    }
    std::cout << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // std::cout << "ncclChannel buffsize " << comm.channels[0] << std::endl;
  int epochs = 10;
  cudaStream_t stream;
  int leastStreamPriority = 0, highestStreamPriority = 0;
  CUDACHECK(cudaDeviceGetStreamPriorityRange(&leastStreamPriority, &highestStreamPriority));
  printf("highestStreamPriority %d\n", highestStreamPriority);
  cudaStreamCreateWithPriority(&stream, cudaStreamDefault, highestStreamPriority);

  cudaStream_t cutlassStream;
  cudaStreamCreateWithPriority(&cutlassStream, cudaStreamDefault, leastStreamPriority);

  cublasHandle_t handle;
  CUBLASCHECK(cublasCreate(&handle));
  CUBLASCHECK(cublasSetStream(handle, stream));
  CUBLASCHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  cublasHandle_t handleWithCutlassStream;
  CUBLASCHECK(cublasCreate(&handleWithCutlassStream));
  CUBLASCHECK(cublasSetStream(handleWithCutlassStream, cutlassStream));
  CUBLASCHECK(cublasSetMathMode(handleWithCutlassStream, CUBLAS_TENSOR_OP_MATH));

  half* dAlpha, *dBeta;
  half alpha = __float2half(1.0);
  CUDACHECK(cudaMalloc(&dAlpha, sizeof(half)));
  CUDACHECK(cudaMemcpy(dAlpha, &alpha, sizeof(half), cudaMemcpyHostToDevice));
  CUDACHECK(cudaMalloc(&dBeta, sizeof(half)));
  half beta = __float2half(0);
  CUDACHECK(cudaMemcpy(dBeta, &beta, sizeof(half), cudaMemcpyHostToDevice));
  CUBLASCHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE));
  CUBLASCHECK(cublasSetPointerMode(handleWithCutlassStream, CUBLAS_POINTER_MODE_DEVICE));

  MPI_Barrier(MPI_COMM_WORLD);

  nChannels = atoi(getenv ("NCCL_MIN_NCHANNELS"));

  #define GPT2_PARAMS
  #ifdef GPT2_PARAMS
    int SEQUENCE_LENGTH = 1024;  

    int BATCH_SIZE[] = {8, 16, 32, 64};
    int HIDDEN_DIMENSIONS[] = {/*345M Model*/ 4096, /*1.2B Model is 1536*/ 4096, /*2.5B Model is 1920*/ 4096, 
                              /*4.2B is 2304*/ 4096};
    int HIDDEN_DIMENSIONS_12CHANNELS[] = {3072, /*345M Model*/ 3072, /*1.2B Model is 1536*/ 3072, /*2.5B Model is 1920*/ 3072, 
                                          /*4.2B is 2304*/ 3072};
    int MODEL_PARALLEL_GPUS[] = {2, 2, 2, 2};
    float MODEL_PARAMS[] = {8.3, 8.3, 8.3, 8.3, 8.3};
  #else
    int SEQUENCE_LENGTH = 2048;  

    int BATCH_SIZE[] = {1, 2, 4, 6};
    int HIDDEN_DIMENSIONS[] = {/*345M Model*/ 12288, /*1.2B Model is 1536*/ 12288, /*2.5B Model is 1920*/ 12288, 12288};
    int HIDDEN_DIMENSIONS_12CHANNELS[] = {/*345M Model*/ 12288, /*1.2B Model is 1536*/ 12288, /*2.5B Model is 1920*/ 12288, 12288};
    int MODEL_PARALLEL_GPUS[] = {2, 2, 2, 2};
    float MODEL_PARAMS[] = {137, 137, 137, 137};
  #endif

  
  for (int model = 0; model < sizeof(HIDDEN_DIMENSIONS)/sizeof(HIDDEN_DIMENSIONS[0]); model++) {
    for (int matMulType = 1; matMulType < 2; matMulType++) {
      int M = BATCH_SIZE[model] * SEQUENCE_LENGTH;
      int N = (nChannels%3 == 0) ? HIDDEN_DIMENSIONS_12CHANNELS[model] : HIDDEN_DIMENSIONS[model];
      int K = N/MODEL_PARALLEL_GPUS[model] * ((matMulType == 0) ? 1 : 4);

      
      if (rank == 0)
        printf("Model Size %.2f B Params , MatMul: [%d, %d] X [%d, %d]\n", MODEL_PARAMS[model], M, K, K, N);

      
      half* m1;
      CUDACHECK(cudaMalloc(&m1, M*K * sizeof(half)));
      // cudaMemRandInt(m1, M*K);
      memset_value(m1, __float2half(1.0f), M*K);
      half* m2;
      CUDACHECK(cudaMalloc(&m2, K*N * sizeof(half)));
      // cudaMemRandInt(m2, K*N);
      memset_value(m2, __float2half(1.0f), K*N);
      half* m1m2;
      CUDACHECK(cudaMalloc(&m1m2,  M*N* sizeof(half)));
      
      half* _m1m2;
      CUDACHECK(cudaMalloc(&_m1m2,  M*N* sizeof(half)));

      half* __m1m2;
      CUDACHECK(cudaMalloc(&__m1m2,  M*N* sizeof(half)));

      MPI_Barrier(MPI_COMM_WORLD);

      float totalTime = 0;
      float cublasTime = 0;
      float allReduceTime = 0;
      float matmulTime = 0;
      
      #define CUBLAS_BASELINE
      #define CUSTOM_BASELINE
      #ifdef CUBLAS_BASELINE
      for(int iter = 0; iter < 110; iter++) {
        if (rank == 0 and iter % 20 == 0)
          printf("iter %d\n", iter);
        cudaEvent_t startpipe, stoppipe;
        float elapsedTimepipe;
        float __allReduceTime = 0.0f, __cublasTime = 0.0f;

        double t1 = getCurrentTime();
        pipe_rowmajorABC(handle, dAlpha, dBeta, m1, m2, m1m2, comms, stream, M, N, K, __allReduceTime, __cublasTime); 

        double t2 = getCurrentTime();
        if (iter >= 10) {
          totalTime += (t2-t1)*1000.0f;
          allReduceTime += __allReduceTime;
          cublasTime += __cublasTime;
        }
        if (iter == 0) 
        { 
          float *hm1 = new float[M*K];
          float *hm2 = new float[N*K];
          float *hm1m2 = new float[M*N];
          
          cudaMemcpyHalfDevice2FloatHost(hm1, m1, M*K);
          cudaMemcpyHalfDevice2FloatHost(hm2, m2, N*K);
          cudaMemcpyHalfDevice2FloatHost(hm1m2, m1m2, M*N);
          if (rank == 0)
            printf("checking results at iter %d \n", iter);
          if (!mpiRef(hm1, hm2, hm1m2, M, N, K, comm_size))
            assert(false);
        }
      }
      MPI_Barrier(MPI_COMM_WORLD);
      if (rank == 0)
      printf("AllReduce+cuBLAS: TotalTime %f ms, AllReduceTime %f ms, cuBLAS Time %f ms\n", totalTime, allReduceTime, cublasTime);
      #endif

      
      memset_value(m1m2, __float2half(0.0f), M*N);
      totalTime = 0.0;
      allReduceTime = 0;
      matmulTime = 0;
      int chunkRows;
      int chunkCols = 512;
      assert(N % chunkCols == 0);
      std::vector<std::vector<std::tuple<int, int, int, int>>> chunkBlocks = getChunkBlocks<half>(rank, M*N, comm_size, rings, M, N, chunkCols, chunkRows) ;

      if (rank == 0 && false) {
        float time = cutlassGeMM(M, N, K, rank, chunkBlocks);

        printf("cutlass GeMM Time: %f\n", time);
      }
      
      MPI_Barrier(MPI_COMM_WORLD);

      
      {
        float cutlassTime = 0.0f;
        allReduceTime = 0.0f;
        //Overlapped AllReduce + CUTLASS
        int length_m = M;
        int length_n = N;
        int length_k = K;
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a((cutlass::half_t*)m1, LayoutInputA::packed(problem_size.mk()));
        cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b((cutlass::half_t*)m2, LayoutInputA::packed(problem_size.kn()));
        cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c((cutlass::half_t*)_m1m2, LayoutInputA::packed(problem_size.mn()));
        cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_d((cutlass::half_t*)m1m2, LayoutInputA::packed(problem_size.mn()));

        // Initialize alpha and beta for dot product computation
        ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
        ElementComputeEpilogue beta = ElementComputeEpilogue(0);

        // Split K dimension into 1 partitions
        int split_k_slices = 1;
                
        //Initialize the memory for thread block to tile map.
        int numTiles = (length_m*length_n)/(ShapeMMAThreadBlock::kMN);
        int* threadBlockToTileMap;
        int* tileIdx;
        int* tileStatusMap;

        CUDACHECK(cudaMalloc(&tileIdx, sizeof(int)));
        CUDACHECK(cudaMemset(tileIdx, 0, sizeof(int)));

        CUDACHECK(cudaMalloc(&threadBlockToTileMap, numTiles * 2 * sizeof(int)));

        //An array of integers for each tile to indicate if tile is waiting (0) or finished (1)
        CUDACHECK(cudaMalloc(&tileStatusMap, numTiles * 4 * sizeof(int)));
        CUDACHECK(cudaMemset(tileStatusMap, 0, numTiles * 4 * sizeof(int)));

        //Create an array of tile order.
        ShapeMMAThreadBlock shape;
        int *tileOrder = new int[numTiles * 2];

        int idx = 0;
        for (int ty = 0; ty < length_m/ShapeMMAThreadBlock::kM; ty++) {
          for (int tx = 0; tx < length_n/ShapeMMAThreadBlock::kN; tx++) {
            tileOrder[idx] = tx;
            tileOrder[idx + 1] = ty;
            idx += 2;
          } 
        }

        std::vector<int> hChunksForTile;
        int maxChunksForTile = 0;

        const int combinedChunks = nChannels;
        
        if (true) {
          idx = 0;
          int chunk = 0;

          std::set<std::pair<int, int>> chunkTBs;
          std::vector<std::pair<int, int>> tileOrderAsPair;
          std::map<int, std::set<int>> tileToChunks;
              int tilesForChunk = 0;

          for (auto channelChunks: chunkBlocks) {
            for (int channel = 0; channel < channelChunks.size(); channel++) {
              auto chunk = channelChunks[channel];
              int cy = std::get<0>(chunk);
              int cx = std::get<1>(chunk);
              int m = std::get<2>(chunk);
              int n = std::get<3>(chunk);

              int chunkIndex = cy/chunkRows * N/chunkCols + cx/chunkCols;

              //For a chunk get all tiles required to obtain this chunk
              int startTy = (cy/ ShapeMMAThreadBlock::kM) * ShapeMMAThreadBlock::kM;

              for (int ty = startTy; ty < min(cy + m, length_m); ty += ShapeMMAThreadBlock::kM) {
                for (int tx = cx; tx < min(cx + n, length_n); tx += ShapeMMAThreadBlock::kN) {
                  int tileIndex = ty/ShapeMMAThreadBlock::kM * (N/ShapeMMAThreadBlock::kN) + tx/ShapeMMAThreadBlock::kN;
                  if (tileToChunks[tileIndex].count(chunkIndex/combinedChunks) == 0) {
                    tileToChunks[tileIndex].insert(chunkIndex/combinedChunks);
                    // if (rank == 0 && cy >= 7920) {
                    //   printf("cy %d cx %d chunkIndex %d\n", cy, cx, chunkIndex);
                    //   tilesForChunk++;
                    // }
                  }

                  
                  // if (chunkIndex == 0) {
                  //   if (rank == 0) 
                  //     printf("1199: %d x %d -> %d x %d -> %d\n", 
                  //            cy, cx, ty/ShapeMMAThreadBlock::kM, tx/ShapeMMAThreadBlock::kN, tileIndex);
                  // }

                  if (chunkTBs.count(std::make_pair(ty,tx)) == 0) {
                    chunkTBs.insert(std::make_pair(ty,tx));
                    // if (rank == 0 && channel == 0) 
                    //   printf("%d x %d -> %d x %d -> %d\n", cy, cx, ty/ShapeMMAThreadBlock::kM, tx/ShapeMMAThreadBlock::kN, tileIndex);
                    
                    tileOrderAsPair.push_back(std::make_pair(tx/ShapeMMAThreadBlock::kN, ty/ShapeMMAThreadBlock::kM));
                  }
                }
              }

            }
          }

          // if (rank == 0) {
          //   printf("rank %d tilesForChunk %d\n", rank, tilesForChunk);
          // }

          for (auto v : tileToChunks) {
            maxChunksForTile = std::max(maxChunksForTile, (int)v.second.size());
          }

          hChunksForTile = std::vector<int>(maxChunksForTile * numTiles, 0);

          for (auto it : tileToChunks) {
            int i = 0;
            for (int c : it.second) {
              hChunksForTile[it.first * maxChunksForTile + i] = c;
              i++;
            }
            for (; i < maxChunksForTile; i++) {
              hChunksForTile[it.first * maxChunksForTile + i] = -1;
            }
          }

          int _idx = 0;
          for (int i = 0; i < tileOrderAsPair.size(); i++) {
            tileOrder[_idx] = tileOrderAsPair[i].second; //Swap because x ("m") is row and y ("n") is column.
            tileOrder[_idx+1] = tileOrderAsPair[i].first;

            // printf("%d %d\n", tileOrder[_idx], tileOrder[_idx + 1]);
            _idx += 2;
            idx += 2;
          }    
        }

        int* chunksForTile;
        
        CUDACHECK(cudaMemcpy(threadBlockToTileMap, tileOrder, numTiles * 2 * sizeof(int), cudaMemcpyHostToDevice));

        CUDACHECK(cudaMalloc(&chunksForTile, hChunksForTile.size() * sizeof(int)));
        CUDACHECK(cudaMemcpy(chunksForTile, &hChunksForTile[0], hChunksForTile.size() * sizeof(int), cudaMemcpyHostToDevice));
        
        // delete[] tileOrder;

        typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                           tensor_a,  // <- reference to matrix A on device
                                           tensor_b,  // <- reference to matrix B on device
                                           tensor_c,  // <- reference to matrix C on device
                                           tensor_d,  // <- reference to matrix D on device
                                           maxChunksForTile,
                                           chunksForTile,
                                           tileIdx,
                                           threadBlockToTileMap,
                                           tileStatusMap,
                                           {alpha, beta},          // <- tuple of alpha and beta
                                           split_k_slices};        // <- k-dimension split factor

        // Using the arguments, query for extra workspace required for matrix multiplication computation
        size_t workspace_size = Gemm::get_workspace_size(arguments);

        // Allocate workspace memory
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

        // Instantiate CUTLASS kernel depending on templates
        Gemm gemm_op;

        // Check the problem size is supported or not 
        cutlass::Status status = gemm_op.can_implement(arguments);
        CUTLASS_CHECK(status);

        status = gemm_op.initialize(arguments, workspace.get());
        CUTLASS_CHECK(status);
          // cudaProfilerStart();
          // CUDACHECK(cudaFuncSetAttribute(dummyKernel<80>,
          //                           cudaFuncAttributeMaxDynamicSharedMemorySize,
          //                           96*1024));
        CUDACHECK(cudaMemset(tileIdx, 0, sizeof(int)));
        CUDACHECK(cudaMemset(tileStatusMap, 0, numTiles * 4 * sizeof(int)));
        float minSampleTime = 10000000.0f;
        float sampleTime;

        for(int iter = 0; iter < 110; iter++) {
          //CUDACHECK(cudaMemset(tileIdx, 0, sizeof(int)));

          // CUDACHECK(cudaMemset(tileStatusMap, 0, numTiles * sizeof(int)));

          if (rank == 0 && iter %20 == 0)
            printf("iter %d\n", iter);
          cudaEvent_t startpipe, stoppipe;
          cudaEvent_t cutlassStartPipe, cutlassStopPipe;
          float elapsedTimepipe, cutlassElapsedTimepipe;
          // MPI_Barrier(MPI_COMM_WORLD);

          CUDACHECK(cudaEventCreate(&startpipe));
          CUDACHECK(cudaEventCreate(&stoppipe));
          CUDACHECK(cudaEventCreate(&cutlassStartPipe));
          CUDACHECK(cudaEventCreate(&cutlassStopPipe));
          CUDACHECK(cudaEventRecord(startpipe, stream));
          CUDACHECK(cudaEventRecord(cutlassStartPipe, cutlassStream));

          double t1 = getCurrentTime();         

          //NCCLCHECK(ncclAllReduceMatrix(m1m2, M*N, M, N, N, ncclHalf, ncclSum, comm, stream));
          NCCLCHECK(ncclAllReduceOverlapMatMul((const void*)m1, (void*)m2, (void*)m1m2, tileStatusMap, M*N, M, N, K, chunkCols, iter, ncclHalf, ncclSum, comms, stream));          

          // dummyKernel<80><<<12, 1024, 96*1024, stream>>>(tileStatusMap, numTiles, iter);

           // First run to check results
          status = gemm_op(iter, cutlassStream);
          CUTLASS_CHECK(status);

          // Wait for kernels to finish
          // CUDACHECK(cudaDeviceSynchronize());

          // CUBLASCHECK(cublasGemmEx(handleWithCutlassStream, CUBLAS_OP_N, CUBLAS_OP_N, 
          // N, M, K, 
          // dAlpha,
          // m2, CUDA_R_16F, N,
          // m1, CUDA_R_16F, K,
          // dBeta, 
          // m1m2, CUDA_R_16F, N,
          // CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
          
          // Check processed order of tiles by cutlass.
          // CUDACHECK(cudaDeviceSynchronize());
          // int* hTileProcessedOrder = new int[numTiles*2];
          // CUDACHECK(cudaMemcpy(hTileProcessedOrder, tileStatusMap + numTiles, 2*numTiles*sizeof(int), cudaMemcpyDeviceToHost));
          // if (true) {
          //   for (int i = 0; i < numTiles; i++) {
          //     if (hTileProcessedOrder[2*i] != tileOrder[2*i]) {
          //       printf("1392: hTileProcessedOrder[%d] = %d, tileOder[%d] = %d\n", i, hTileProcessedOrder[2*i], i, tileOrder[2*i]);
                
          //     }
          //     if (hTileProcessedOrder[2*i + 1] != tileOrder[2*i + 1]) {
          //       printf("1396: hTileProcessedOrder[%d] = %d\n", i, hTileProcessedOrder[i]);
          //       break;
          //     }
          //   }
          // } 



          CUDACHECK(cudaEventRecord(cutlassStopPipe, cutlassStream));
          CUDACHECK(cudaEventSynchronize(cutlassStopPipe));
          CUDACHECK(cudaEventElapsedTime(&cutlassElapsedTimepipe, cutlassStartPipe,cutlassStopPipe));
          // printf("cutlassElapsedTimepipe %f\n", cutlassElapsedTimepipe);
          CUDACHECK(cudaEventRecord(stoppipe, stream));
          CUDACHECK(cudaEventSynchronize(stoppipe));
          double t2 = getCurrentTime();

          CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
          CUDACHECK(cudaEventElapsedTime(&cutlassElapsedTimepipe, cutlassStartPipe,cutlassStopPipe));
          
          if (iter >= 10) {
            totalTime += (t2-t1)*1000.0f;
            allReduceTime += elapsedTimepipe;
            cutlassTime += cutlassElapsedTimepipe;
            sampleTime += (t2-t1)*1000.0f;

            if (iter > 10 && iter % 10 == 0) {
              minSampleTime = std::min(minSampleTime, sampleTime*10);
              sampleTime = 0;//(t2-t1)*1000.0f;
            }
          }
          if (iter == 0) 
          { 
            MPI_Barrier(MPI_COMM_WORLD);
            float *hm1 = new float[M*K];
            float *hm2 = new float[N*K];
            float *hm1m2 = new float[M*N];
            
            cudaMemcpyHalfDevice2FloatHost(hm1, m1, M*K);
            cudaMemcpyHalfDevice2FloatHost(hm2, m2, N*K);
            cudaMemcpyHalfDevice2FloatHost(hm1m2, m1m2, M*N);
            if (rank == 0)
              printf("checking results at iter %d %d\n", iter, rank);
            if (!mpiRef(hm1, hm2, hm1m2, M, N, K, comm_size, rank))
              assert(false);
          }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
          printf("rank %d Overlapped(AllReduce, cutlass) Time: %f ms cutlass: %f ms, allreduceTime: %f ms, minSampleTime: %f ms\n", rank, totalTime, cutlassTime, allReduceTime, minSampleTime);
        
        // printf("rank %d cutlass %f\n", rank, cutlassTime);
      }
    }
  }
}