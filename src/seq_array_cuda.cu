#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// CUDA kernel to add elements of two arrays
__global__ void add(int n, float *x, float *y, float *z) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    z[index] = x[index] + y[index];
  }
}

// Function to initialize array with random values
void initialize(float *array, int n) {
  srand(time(0));
  for (int i = 0; i < n; i++) {
    array[i] = rand() / (float)RAND_MAX;
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    fprintf(stderr, "Usage: %s <input size> <grid size> <block size>\n",
            argv[0]);
    return 1;
  }

  int N = atoi(argv[1]);         // input size
  int gridSize = atoi(argv[2]);  // grid size
  int blockSize = atoi(argv[3]); // block size

  // Allocate memory for arrays on the host
  float *x = (float *)malloc(N * sizeof(float));
  float *y = (float *)malloc(N * sizeof(float));
  float *z = (float *)malloc(N * sizeof(float)); // output array

  // Initialize arrays with random values
  initialize(x, N);
  initialize(y, N);

  // Allocate memory for arrays on the device
  float *d_x, *d_y, *d_z;
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));
  cudaMalloc(&d_z, N * sizeof(float));

  // Copy input arrays from host to device
  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch the CUDA kernel
  add<<<gridSize, blockSize>>>(N, d_x, d_y, d_z);

  // Copy the result array from device to host
  cudaMemcpy(z, d_z, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Free memory on both host and device
  free(x);
  free(y);
  free(z);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);

  return 0;
}
