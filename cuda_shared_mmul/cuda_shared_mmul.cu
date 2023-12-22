#include "./cuda_shared_mmul.cuh"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "device_launch_parameters.h"
#include "./../type.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "./../cublas_utils.h"

__global__ void gpu_matrix_mult(T* a, T* b, T* c, int m, int n, int k)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if (col < k && row < m)
    {
        for (int i = 0; i < n; i++)
        {
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

__global__ void gpu_square_matrix_mult(int* d_a, int* d_b, int* d_result, int n)
{
    __shared__ int tile_a[BLOCKSIZE][BLOCKSIZE];
    __shared__ int tile_b[BLOCKSIZE][BLOCKSIZE];

    int row = blockIdx.y * BLOCKSIZE + threadIdx.y;
    int col = blockIdx.x * BLOCKSIZE + threadIdx.x;
    int tmp = 0;
    int idx;

    for (int sub = 0; sub < gridDim.x; ++sub)
    {
        idx = row * n + sub * BLOCKSIZE + threadIdx.x;
        if (idx >= n * n)
        {
            // n may not divisible by BLOCKSIZE
            tile_a[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_a[threadIdx.y][threadIdx.x] = d_a[idx];
        }

        idx = (sub * BLOCKSIZE + threadIdx.y) * n + col;
        if (idx >= n * n)
        {
            tile_b[threadIdx.y][threadIdx.x] = 0;
        }
        else
        {
            tile_b[threadIdx.y][threadIdx.x] = d_b[idx];
        }
        __syncthreads();

        for (int k = 0; k < BLOCKSIZE; ++k)
        {
            tmp += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < n && col < n)
    {
        d_result[row * n + col] = tmp;
    }
}

T* executeGPU(int n, int m, int k, T* A, T* B) {
    T* h_A = A, * h_B = B, * h_C = (T*)malloc(sizeof(T) * m * n);
    int lda = n, ldb = k, ldc = n;

    if (n <= SIZE_LIMIT && k <= SIZE_LIMIT && n <= SIZE_LIMIT) {
        printf("A\n");
        print_matrix<T>(n, k, h_A, lda);
        printf("=====\n");

        printf("B\n");
        print_matrix<T>(k, m, h_B, ldb);
        printf("=====\n");
    }

    const T alpha = 1.0;
    const T beta = 0.0;

    T* d_A = nullptr;
    T* d_B = nullptr;
    T* d_C = nullptr;
    CUDA_CHECK(cudaSetDevice(0));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc((void**)(&d_A), sizeof(T) * n * k));
    CUDA_CHECK(cudaMalloc((void**)(&d_B), sizeof(T) * k * m));
    CUDA_CHECK(cudaMalloc((void**)(&d_C), sizeof(T) * n * m));

    // Высчитываем с момента копирования
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeof(T) * n * k, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeof(T) * k * m, cudaMemcpyHostToDevice));

    /* step 3: compute */
    unsigned int grid_rows = (m + BLOCKSIZE - 1) / BLOCKSIZE;
    unsigned int grid_cols = (k + BLOCKSIZE - 1) / BLOCKSIZE;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE);
    gpu_matrix_mult << <dimGrid, dimBlock >> > (d_A, d_B, d_C, m, n, k);

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeof(T) * n * m, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Время выполнения GPU: %f мс.\n", milliseconds);

    /*
     *   C = | 23.0 | 31.0 |
     *       | 34.0 | 46.0 |
     */
    if (n * m <= SIZE_LIMIT * SIZE_LIMIT) {
        printf("C\n");
        print_matrix(n, m, h_C, ldc);
        printf("=====\n");
    }

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUDA_CHECK(cudaDeviceReset());
}