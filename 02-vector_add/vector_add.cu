#include<stdio.h>

/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
 __global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

int main(int argc,char **argv)
{
  cudaError_t err = cudaSuccess;

  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("vector add %d elements\n", numElements);

  float *h_A = (float*)malloc(size);
  float *h_B = (float*)malloc(size);
  float *h_C = (float*)malloc(size);

  if (h_A == NULL || h_B == NULL || h_C == NULL) {
    printf("Error: Failed to allocate host vectors!\n");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < numElements; i++) {
    h_A[i] = rand() / (float)RAND_MAX;
    h_B[i] = rand() / (float)RAND_MAX;
  }

  float *d_A = NULL;
  float *d_B = NULL;
  float *d_C = NULL;

  err = cudaMalloc((void **)&d_A, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to alloc device vector A, error(%s)\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **)&d_B, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to alloc device vector B, error(%s)\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMalloc((void **)&d_C, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to alloc device vector C, error(%s)\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy A from host to device, error(%s)\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy B from host to device, error(%s)\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Launch cuda kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("Copy output data from the CUDA device to the host memory\n");
  err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr,
            "Failed to copy vector C from device to host (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // verify
  for (int i = 0; i < numElements; i++) {
    if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }
  printf("Test passed!\n");

  // free resoruce
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

  free(h_A);
  free(h_B);
  free(h_C);

  return 0;
}
