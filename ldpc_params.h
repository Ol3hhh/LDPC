#pragma once
#include <cmath>
#include <cuda_runtime.h>

const int NUM_VN = 12;
const int NUM_CN = 6;
const int MAX_DEG_C = 6;
const int MAX_DEG_V = 3;
const int MAX_ITERS = 15;
const float ALPHA = 0.75f;

const int h_H[NUM_CN][NUM_VN] = {
    {1,1,0,1,0,0,0,0,0,0,1,0},
    {0,1,1,0,1,0,0,0,1,0,0,0},
    {1,0,1,0,0,1,0,0,0,0,0,1},
    {0,0,0,1,1,0,1,0,0,0,1,0},
    {0,0,0,0,0,1,0,1,1,1,0,0},
    {0,0,0,0,1,0,1,0,0,1,0,1}
};

__device__ __host__ inline float signf(float x) { 
    return (x >= 0.0f) ? 1.0f : -1.0f; 
}
