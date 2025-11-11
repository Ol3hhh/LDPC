#include "decoder_gpu.h"
#include "ldpc_params.h"

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <numeric>

using namespace std;

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d (%s): %s\n", file, line, func, cudaGetErrorString(err));
        exit(1);
    }
}

__global__ void kernel_vnu(
    const int* d_col_ptr, const int* d_row_idx,
    const float* d_llr, const float* d_M_cv, float* d_M_vc)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= NUM_VN) return;

    int start = d_col_ptr[v];
    int end = d_col_ptr[v+1];
    
    float total_sum = d_llr[v];
    for(int e = start; e < end; ++e){
        int c = d_row_idx[e];
        total_sum += d_M_cv[c*NUM_VN + v];
    }

    for(int e = start; e < end; ++e){
        int c = d_row_idx[e];
        d_M_vc[c*NUM_VN + v] = total_sum - d_M_cv[c*NUM_VN + v];
    }
}

__global__ void kernel_cnu(
    const int* d_row_ptr, const int* d_col_idx,
    const float* d_M_vc, float* d_M_cv)
{
    extern __shared__ char s_mem[];
    float* s_vals = (float*)s_mem; 
    float& s_min1      = *(float*)(s_mem + sizeof(float) * MAX_DEG_C);
    float& s_min2      = *(float*)(s_mem + sizeof(float) * (MAX_DEG_C + 1));
    float& s_sign_all  = *(float*)(s_mem + sizeof(float) * (MAX_DEG_C + 2));
    float& s_idx_min_f = *(float*)(s_mem + sizeof(float) * (MAX_DEG_C + 3));

    int c = blockIdx.x;
    int start = d_row_ptr[c];
    int end = d_row_ptr[c+1];
    int deg = end - start;
    int tid = threadIdx.x;

    if(tid < deg){
        int v = d_col_idx[start + tid];
        s_vals[tid] = d_M_vc[c*NUM_VN + v];
    }
    __syncthreads();

    if(tid == 0){
        float min1=1e9f, min2=1e9f;
        int idx_min = -1;
        float sign_all=1.0f;
        
        for(int i=0; i<deg; i++){
            float val = s_vals[i];
            sign_all *= signf(val);
            float av = fabsf(val);
            if(av < min1){
                min2 = min1; min1 = av; idx_min = i;
            } else if(av < min2){
                min2 = av;
            }
        }

        s_min1 = min1;
        s_min2 = min2;
        s_idx_min_f = (float)idx_min;
        s_sign_all = sign_all;
    }
    
    __syncthreads();

    if(tid < deg){
        float my_val = s_vals[tid];
        float s = s_sign_all * signf(my_val);
        float minval = (tid == (int)s_idx_min_f) ? s_min2 : s_min1; 
        
        float msg = ALPHA * s * minval;
        int v = d_col_idx[start + tid];
        d_M_cv[c*NUM_VN + v] = msg;
    }
}

__global__ void kernel_decision(
    const int* d_col_ptr, const int* d_row_idx,
    const float* d_llr, const float* d_M_cv, int* d_out)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if(v >= NUM_VN) return;
    int start = d_col_ptr[v], end = d_col_ptr[v+1];
    float sum = d_llr[v];
    for(int e = start; e < end; ++e){
        int c = d_row_idx[e];
        sum += d_M_cv[c*NUM_VN + v];
    }
    d_out[v] = (sum >= 0.0f) ? 0 : 1;
}

void run_ldpc_decode_gpu(
    const std::vector<float>& h_llr, 
    std::vector<int>& h_gpu_out)
{
    vector<vector<int>> N_c(NUM_CN), N_v(NUM_VN);
    for(int c=0;c<NUM_CN;c++)
        for(int v=0;v<NUM_VN;v++)
            if(h_H[c][v]){
                N_c[c].push_back(v);
                N_v[v].push_back(c);
            }

    vector<int> h_row_ptr(NUM_CN+1,0);
    vector<int> h_col_idx;
    for(int c=0;c<NUM_CN;c++){
        h_row_ptr[c+1]=h_row_ptr[c]+N_c[c].size();
        for(int v: N_c[c]) h_col_idx.push_back(v);
    }

    vector<int> h_col_ptr(NUM_VN+1,0);
    vector<int> h_row_idx;
    for(int v=0;v<NUM_VN;v++){
        h_col_ptr[v+1]=h_col_ptr[v]+N_v[v].size();
        for(int c: N_v[v]) h_row_idx.push_back(c);
    }

    int *d_row_ptr, *d_col_idx, *d_col_ptr, *d_row_idx;
    float *d_llr, *d_M_vc, *d_M_cv;
    int *d_out;
    CHECK_CUDA_ERROR(cudaMalloc(&d_row_ptr,sizeof(int)*(NUM_CN+1)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_col_idx,sizeof(int)*h_col_idx.size()));
    CHECK_CUDA_ERROR(cudaMalloc(&d_col_ptr,sizeof(int)*(NUM_VN+1)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_row_idx,sizeof(int)*h_row_idx.size()));
    CHECK_CUDA_ERROR(cudaMalloc(&d_llr,sizeof(float)*NUM_VN));
    CHECK_CUDA_ERROR(cudaMalloc(&d_M_vc,sizeof(float)*NUM_CN*NUM_VN));
    CHECK_CUDA_ERROR(cudaMalloc(&d_M_cv,sizeof(float)*NUM_CN*NUM_VN));
    CHECK_CUDA_ERROR(cudaMalloc(&d_out,sizeof(int)*NUM_VN));

    CHECK_CUDA_ERROR(cudaMemcpy(d_row_ptr,h_row_ptr.data(),sizeof(int)*(NUM_CN+1),cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_col_idx,h_col_idx.data(),sizeof(int)*h_col_idx.size(),cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_col_ptr,h_col_ptr.data(),sizeof(int)*(NUM_VN+1),cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_row_idx,h_row_idx.data(),sizeof(int)*h_row_idx.size(),cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_llr,h_llr.data(),sizeof(float)*NUM_VN,cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemset(d_M_vc,0,sizeof(float)*NUM_CN*NUM_VN));
    CHECK_CUDA_ERROR(cudaMemset(d_M_cv,0,sizeof(float)*NUM_CN*NUM_VN));

    dim3 vBlocks((NUM_VN+31)/32); dim3 vThreads(32);
    dim3 cBlocks(NUM_CN); dim3 cThreads(MAX_DEG_C);

    const int total_shared_mem_size = sizeof(float) * (MAX_DEG_C + 4);

    for(int it=0; it<MAX_ITERS; ++it){
        kernel_vnu<<<vBlocks,vThreads>>>(d_col_ptr,d_row_idx,d_llr,d_M_cv,d_M_vc);
        CHECK_CUDA_ERROR(cudaGetLastError()); 
        
        kernel_cnu<<<cBlocks,cThreads,total_shared_mem_size>>>(d_row_ptr,d_col_idx,d_M_vc,d_M_cv);
        CHECK_CUDA_ERROR(cudaGetLastError());
    }
    kernel_decision<<<vBlocks,vThreads>>>(d_col_ptr,d_row_idx,d_llr,d_M_cv,d_out);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    CHECK_CUDA_ERROR(cudaMemcpy(h_gpu_out.data(),d_out,sizeof(int)*NUM_VN,cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    cudaFree(d_row_ptr); cudaFree(d_col_idx); cudaFree(d_col_ptr); cudaFree(d_row_idx);
    cudaFree(d_llr); cudaFree(d_M_vc); cudaFree(d_M_cv); cudaFree(d_out);
}
