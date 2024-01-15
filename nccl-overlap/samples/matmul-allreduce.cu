#include "header.h"
#include "cutlass-matmul.h"
#include <cuda_profiler_api.h>

#include <map>

#define MAX_CHANNELS 80

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
  // getNCCLRings(&comms, rings, ringLength, nChannels);
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
}
