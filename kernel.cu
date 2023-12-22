#include <Windows.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>

#include <stdio.h>
#include <iostream>
#include "./cublas_gemm/cublas_gemm.cuh"
#include "./cpu_mmul/cpu_mmul.cuh"


void test() {
    int n = 2, k = 3, m = 2;
    std::vector<T> matrix1 = { 1, 2, 3, 4, 5, 6 };
    std::vector<T> matrix2 = { 7, 8, 9, 10, 11, 12 };
    executeCPU(n, m, k, matrix1.data(), matrix2.data());
    executeCublas(n, m, k, matrix1.data(), matrix2.data());

    n = 3, k = 3, m = 3;
    std::vector<T> matrix3 = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<T> matrix4 = { 10, 11, 12,13,14,15,16,17,18 };
    executeCPU(n, m, k, matrix3.data(), matrix4.data());
    executeCublas(n, m, k, matrix3.data(), matrix4.data());

    n = 3, k = 3, m = 1;
    std::vector<T> matrix5 = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<T> matrix6 = { 10, 13,16 };
    executeCPU(n, m, k, matrix5.data(), matrix6.data());
    executeCublas(n, m, k, matrix5.data(), matrix6.data());

    n = 1, k = 3, m = 3;
    std::vector<T> matrix7 = { 1, 2, 3 };
    std::vector<T> matrix8 = { 10, 11, 12,13,14,15,16,17,18 };
    executeCPU(n, m, k, matrix7.data(), matrix8.data());
    executeCublas(n, m, k, matrix7.data(), matrix8.data());

    n = 2, k = 3, m = 1;
    std::vector<T> matrix9 = { 1,2,4,5,7,8 };
    std::vector<T> matrix10 = { 10,13,16 };
    executeCPU(n, m, k, matrix9.data(), matrix10.data());
    executeCublas(n, m, k, matrix9.data(), matrix10.data());

}

// A = (n x k); B = (k x m); C = (n x m);
int main() {
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);

    /*int n, m, k;
    std::cout << "n: ";
    std::cin >> n;
    std::cout << "m: ";
    std::cin >> m;
    std::cout << "k: ";
    std::cin >> k;
    T *A = generate_random_int(n, k, 10);
    T *B = generate_random_int(k, m, 10);
    executeCublas(n, m, k, &A, &B);*/
    test();
    return 0;
}