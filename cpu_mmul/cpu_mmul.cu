#include "cpu_mmul.cuh"
#include <cstdlib>
#include <iostream>

void printMatrix(int n, int m, T* A) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			std::printf("%3.0f ", (A)[i + j * n]);
		}
		std::printf("\n");
	}
};

void executeCPU(int n, int m, int k, T* A, T* B) {
	T* C = (T*)malloc(sizeof(T) * n * m);
	std::printf("A\n");
	printMatrix(n, k, A);
	std::printf("=====\n");
	std::printf("B\n");
	printMatrix(k, m, B);
	std::printf("=====\n");

	for (size_t i = 0; i < n; i++)
	{
		for (size_t j = 0; j < m; j++)
		{
			C[i * m + j] = 0;
			for (size_t l = 0; l < k; l++)
			{
				int inda = i * k + l;
				int indb = l * m + j;
				double a = (A)[inda];
				double b = (B)[indb];
				C[i * m + j] += a * b;
			}
		}
	}

	printMatrix(n, m, C);
};