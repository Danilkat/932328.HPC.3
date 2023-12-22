#include "cpu_mmul.cuh"
#include <cstdlib>
#include <iostream>
#include <cublas_v2.h>
#include "./../cublas_utils.h"

T* executeCPU(int n, int m, int k, T* A, T* B, double* time) {
	T* C = (T*)malloc(sizeof(T) * n * m);
	if (n <= SIZE_LIMIT && k <= SIZE_LIMIT && n <= SIZE_LIMIT) {
		std::printf("A\n");
		print_matrix(n, k, A, n);
		std::printf("=====\n");
		std::printf("B\n");
		print_matrix(k, m, B, k);
		std::printf("=====\n");
	}
	clock_t start_time = clock();
	for (size_t j = 0; j < m; j++)
	{
		for (size_t i = 0; i < n; i++)
		{
			C[j * n + i] = 0;
			for (size_t l = 0; l < k; l++)
			{
				int inda = l * n + i;
				int indb = j * k + l;
				double a = (A)[inda];
				double b = (B)[indb];
				C[j * n + i] += a * b;
			}
		}
	}
	clock_t end_time = clock();
	if (n * m <= SIZE_LIMIT * SIZE_LIMIT * OUTPUT_MULTIPLIER) {
		std::printf("C\n");
		print_matrix(n, m, C, n);
		std::printf("=====\n");
	}
	int search_time = end_time - start_time;
	printf("Время выполнения CPU: %d мс. (%2.3f с.)\n\n", search_time, (double)search_time / CLOCKS_PER_SEC);
	*time = ((double)search_time);
	return C;
};