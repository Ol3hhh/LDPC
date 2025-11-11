#pragma once
#include <vector>

void run_ldpc_decode_gpu(
    const std::vector<float>& h_llr, 
    std::vector<int>& h_gpu_out
);
