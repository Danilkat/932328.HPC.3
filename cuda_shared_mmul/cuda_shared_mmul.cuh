#pragma once
#include "./../type.h"
#define BLOCKSIZE 32
#define uint unsigned int
T* executeGPU(int n, int m, int k, T* A, T* B);