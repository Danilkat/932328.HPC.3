#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../cublas_utils.h"
#include "cublas_gemm.cuh"

T* generate_random_int(int m, int n, int sigma) {
    T* A = (T*)malloc(sizeof(T) * m * n);
    int drop = 0;
    generate_random_matrix<T>(m, n, &A, &drop);
    for (size_t i = 0; i < m * n; i++)
    {
        T val = round(A[i] * sigma);
        A[i] = (abs(val) > std::numeric_limits<T>::epsilon()) ? val : 0.0;
    };
    for (size_t i = 0; i < m*n; i++)
    {
        std::printf("%3.0f ", A[i]);
    }
    std::printf("\n");
    return A;
};

void executeCublas(int n, int m, int k, double **A, double **B) {
	cublasHandle_t cublasH = NULL;
	cudaStream_t stream = NULL;
	T* h_A = *A, * h_B = *B, * h_C = (T*)malloc(sizeof(T) * n * m);
	int lda = k > n ? n : k, ldb = k > m ? m : k, ldc = n;

	if (n <= 10 && k <= 10 && n <= 10) {
		printf("A\n");
		print_matrix<T>(n, k, h_A, lda);
		printf("=====\n");

		printf("B\n");
		print_matrix<T>(k, m, h_B, ldb);
		printf("=====\n");
	}

	const double alpha = 1.0;
	const double beta = 0.0;

	T* d_A = nullptr;
	T* d_B = nullptr;
	T* d_C = nullptr;

	cublasOperation_t transa = n > k ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t transb = m >= k ? CUBLAS_OP_N : CUBLAS_OP_T;

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(cublasCreate(&cublasH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUBLAS_CHECK(cublasSetStream(cublasH, stream));

    /* step 2: copy data to device */
    CUDA_CHECK(cudaMalloc((void**)(&d_A), sizeof(T) * n * k));
    CUDA_CHECK(cudaMalloc((void**)(&d_B), sizeof(T) * k * m));
    CUDA_CHECK(cudaMalloc((void**)(&d_C), sizeof(T) * n * m));

    // Высчитываем с момента копирования
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A, sizeof(T) * n * k, cudaMemcpyHostToDevice,
        stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B, sizeof(T) * k * m, cudaMemcpyHostToDevice,
        stream));

    /* step 3: compute */
    CUBLAS_CHECK(
    cublasDgemm(cublasH, transa, transb, n, m, k, &alpha, d_A, lda, d_B, ldb, &beta, d_C, ldc)
        );

    /* step 4: copy data to host */
    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C, sizeof(T) * n * m, cudaMemcpyDeviceToHost,
        stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Время выполнения GPU: %f мс.\n", milliseconds);

    /*
     *   C = | 23.0 | 31.0 |
     *       | 34.0 | 46.0 |
     */
    if (n * m <= 100) {
        printf("C\n");
        print_matrix(n, m, h_C, ldc);
        printf("=====\n");
    }

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    CUBLAS_CHECK(cublasDestroy(cublasH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());
}