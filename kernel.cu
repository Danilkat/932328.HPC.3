#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include "./cublas_gemm/cublas_gemm.cuh"
#include "./cpu_mmul/cpu_mmul.cuh"


void testCPU() {
    int n = 2, k = 3, m = 2;
    T matrix1[] = { 1, 2, 3, 4, 5, 6 };
    T matrix2[] = { 7, 8, 9, 10, 11, 12 };
    executeCPU(n, m, k, matrix1, matrix2);

    n = 3, k = 3, m = 3;
    T matrix3[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    T matrix4[] = { 10, 11, 12,13,14,15,16,17,18 };
    executeCPU(n, m, k, matrix3, matrix4);

    n = 3, k = 3, m = 1;
    T matrix5[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    T matrix6[] = { 10, 13,16 };
    executeCPU(n, m, k, matrix5, matrix6);

    n = 1, k = 3, m = 3;
    T matrix7[] = { 1, 2, 3 };
    T matrix8[] = { 10, 11, 12,13,14,15,16,17,18 };
    executeCPU(n, m, k, matrix7, matrix8);

}

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
    executeCublas(n, m, k, &A, &B);
    executeCPU(n, m, k, A, B);
    //testCPU();
    return 0;
}