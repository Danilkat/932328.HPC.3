#pragma once
#include "./../type.h"

T* executeCublas(int n, int m, int k, T* A, T* B, double* time);
T* generate_random_int(int m, int n, int sigma);