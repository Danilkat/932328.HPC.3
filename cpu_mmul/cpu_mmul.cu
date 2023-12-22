#include "cpu_mmul.cuh"
#include <cstdlib>
#include <iostream>

void printMatrix(int n, int m, T* A) {clock_t start_time = clock();
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			std::printf("%3.0f ", (A)[i + j * n]);
		}
		std::printf("\n");
	}
};

T* executeCPU(int n, int m, int k, T* A, T* B) {
	T* C = (T*)malloc(sizeof(T) * n * m);
	if (n <= SIZE_LIMIT && k <= SIZE_LIMIT && n <= SIZE_LIMIT) {
		std::printf("A\n");
		printMatrix(n, k, A);
		std::printf("=====\n");
		std::printf("B\n");
		printMatrix(k, m, B);
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
		printMatrix(n, m, C);
		std::printf("=====\n");
	}
	double search_time = end_time - start_time;
	printf("Время выполнения CPU: %d мс.\n\n", search_time);
	return C;
};