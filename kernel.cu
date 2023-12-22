#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "./cublas_gemm/cublas_gemm.cuh"
#include "./cpu_mmul/cpu_mmul.cuh"

// A = (n x k); B = (k x m); C = (n x m);
int main() {
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);

    int n, m, k;
    std::cout << "n: ";
    std::cin >> n;
    std::cout << "m: ";
    std::cin >> m;
    std::cout << "k: ";
    std::cin >> k;
    T *A = generate_random_int(n, k, 10);
    T *B = generate_random_int(k, m, 10);
    //executeCublas(n, m, k, &A, &B);
    //executeCPU(n, m, k, &A, &B);
    testCPU();
    return 0;
}

void testCPU() {
    int n = 2, k = 3, m = 2;
    T matrix1[] = { 1, 2, 3, 4, 5, 6 };
    T matrix2[] = { 7, 8, 9, 10, 11, 12 };
    executeCPU(n, m, k, matrix1, matrix2);
}