#include <stdio.h>
#include <stdlib.h>

// CUDA kernel to add elements of two matrices
__global__ void add(int n, float *x, float *y, float *z) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n * n; i += stride)
    z[i] = x[i] + y[i];
}

// Function to initialize matrix with random numbers
void initialize(float *a, int n) {
  for (int i = 0; i < n * n; i++)
    a[i] = rand() / (float)RAND_MAX;
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <matrix size> <grid size> <block size>\n",
            argv[0]);
    return 1;
  }

  int N = atoi(argv[1]);         // matrix size
  int gridSize = atoi(argv[2]);  // grid size
  int blockSize = atoi(argv[3]); // block size

  // Allocate memory for matrices on the host
  float *x = (float *)malloc(N * N * sizeof(float));
  float *y = (float *)malloc(N * N * sizeof(float));
  float *z = (float *)malloc(N * N * sizeof(float)); // output matrix

  // Initialize matrices with random values
  initialize(x, N);
  initialize(y, N);

  // Allocate memory for matrices on the device
  float *d_x, *d_y, *d_z;
  cudaMalloc(&d_x, N * N * sizeof(float));
  cudaMalloc(&d_y, N * N * sizeof(float));
  cudaMalloc(&d_z, N * N * sizeof(float));

  // Copy input matrices from host to device
  cudaMemcpy(d_x, x, N * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the CUDA kernel
  add<<<gridSize, blockSize>>>(N, d_x, d_y, d_z);

  // Copy the result matrix from device to host
  cudaMemcpy(z, d_z, N * N * sizeof(float), cudaMemcpyDeviceToHost);

  // Free memory on both host and device
  free(x);
  free(y);
  free(z);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  return 0;
}
