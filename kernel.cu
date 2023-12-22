#include <Windows.h>
#include <vector>
#include <stdio.h>
#include <iostream>
#include "./cublas_gemm/cublas_gemm.cuh"
#include "./cpu_mmul/cpu_mmul.cuh"
#include "./cuda_shared_mmul/cuda_shared_mmul.cuh"
#include "./type.h"

void test() {
    int n = 2, k = 3, m = 2;
    std::vector<T> matrix1 = { 1, 2, 3, 4, 5, 6 };
    std::vector<T> matrix2 = { 7, 8, 9, 10, 11, 12 };
    executeCPU(n, m, k, matrix1.data(), matrix2.data(), nullptr);
    executeCublas(n, m, k, matrix1.data(), matrix2.data(), nullptr);
    executeGPU(n, m, k, matrix1.data(), matrix2.data(), nullptr);

    n = 3, k = 3, m = 3;
    std::vector<T> matrix3 = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<T> matrix4 = { 10, 11, 12,13,14,15,16,17,18 };
    executeCPU(n, m, k, matrix3.data(), matrix4.data(), nullptr);
    executeCublas(n, m, k, matrix3.data(), matrix4.data(), nullptr);
    executeGPU(n, m, k, matrix3.data(), matrix4.data(), nullptr);

    n = 3, k = 3, m = 1;
    std::vector<T> matrix5 = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<T> matrix6 = { 10, 13,16 };
    executeCPU(n, m, k, matrix5.data(), matrix6.data(), nullptr);
    executeCublas(n, m, k, matrix5.data(), matrix6.data(), nullptr);
    executeGPU(n, m, k, matrix5.data(), matrix6.data(), nullptr);

    n = 1, k = 3, m = 3;
    std::vector<T> matrix7 = { 1, 2, 3 };
    std::vector<T> matrix8 = { 10, 11, 12,13,14,15,16,17,18 };
    executeCPU(n, m, k, matrix7.data(), matrix8.data(), nullptr);
    executeCublas(n, m, k, matrix7.data(), matrix8.data(), nullptr);
    executeGPU(n, m, k, matrix7.data(), matrix8.data(), nullptr);

    n = 2, k = 3, m = 1;
    std::vector<T> matrix9 = { 1,2,4,5,7,8 };
    std::vector<T> matrix10 = { 10,13,16 };
    executeCPU(n, m, k, matrix9.data(), matrix10.data(), nullptr);
    executeCublas(n, m, k, matrix9.data(), matrix10.data(), nullptr);
    executeGPU(n, m, k, matrix9.data(), matrix10.data(), nullptr);

}

void assertMatrices(T* A, T* B, int n, int m, char* labelA, char* labelB) {
    bool is_right = true;
    for (size_t i = 0; i < n * m; i++)
    {
        if (A[i] != B[i]) {
            printf("Index %d. %s: %d. %s: %d ", i, labelA, A[i], labelB, B[i]);
            printf("FAILED \n");
            is_right = false;
        }
    }
    printf("Результаты между %s и %s %s\n", labelA, labelB, (is_right ? "сошлись" : "не сошлись"));
};

void compareTime(double time1, double time2, char* labelA, char* labelB) {
    double diff = time1 > time2 ? (double)time1 / time2 : (double)time2 / time1;
    char* bestPerf = time1 > time2 ? labelB : labelA;
    char* worsePerf = time1 > time2 ? labelA : labelB;
    printf("%s выполнил умножение быстрее чем %s в %0.3f раз\n", bestPerf, worsePerf, diff);
};
// A = (n x k); B = (k x m); C = (n x m);
int main() {
    SetConsoleCP(1251);
    SetConsoleOutputCP(1251);
    //test();
    int n, m, k;
    std::cout << "n: ";
    std::cin >> n;
    std::cout << "m: ";
    std::cin >> m;
    std::cout << "k: ";
    std::cin >> k;
   /* T *A = generate_random_int(n, k, 10);
    T *B = generate_random_int(k, m, 10);*/
    double timeGPU = 0, timeCUBLAS = 0, timeCPU = 0;

    printf("...начинаем генерацию входных матриц\n");
    T* A = generate_random_int(n, k, 10);
    printf("...сгенерировали матрицу А\n");
    T* B = generate_random_int(k, m, 10);
    printf("...сгенерировали матрицу Б, закончили генреацию\n");

    printf("...выполняем умножение на процессоре\n");
    T* rCPU = executeCPU(n, m, k, A, B, &timeCPU);
    printf("...выполняем умножение используя CUBLAS\n");
    T* rCublas = executeCublas(n, m, k, A, B, &timeCUBLAS);
    printf("...выполняем умножение используя CUDA\n");
    T* rGPU = executeGPU(n, m, k, A, B, &timeGPU);

    printf("...проверяем результаты друг с другом\n");
    assertMatrices(rCPU, rCublas, n, m, "CPU", "CUBLAS");
    assertMatrices(rCPU, rGPU, n, m, "CPU", "GPU");
    assertMatrices(rCublas, rGPU, n, m, "CUBLAS", "GPU");

    printf("\n...сравниваем время\n");
    compareTime(timeCPU, timeCUBLAS, "CPU", "CUBLAS");
    compareTime(timeCPU, timeGPU, "CPU", "GPU");
    compareTime(timeGPU, timeCUBLAS, "GPU", "CUBLAS");
    return 0;
}