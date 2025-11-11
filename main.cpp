#include <cstdio>
#include <cstdlib>
#include <vector>
#include <iostream>
#include "ldpc_params.h"
#include "decoder_gpu.h"

using namespace std;

void cpu_ldpc_decode(const vector<float>& llr, vector<int>& decoded)
{
    vector<vector<int>> N_v(NUM_VN), N_c(NUM_CN);

    for (int c = 0; c < NUM_CN; c++) {
        for (int v = 0; v < NUM_VN; v++) {
            if (h_H[c][v] != 0) {
                N_v[v].push_back(c);
                N_c[c].push_back(v);
            }
        }
    }

    vector<vector<float>> M_cv(NUM_CN, vector<float>(NUM_VN, 0.0f));
    vector<vector<float>> M_vc(NUM_CN, vector<float>(NUM_VN, 0.0f));

    for (int iter = 0; iter < MAX_ITERS; iter++) {

        // Variable -> Check
        for (int v = 0; v < NUM_VN; v++) {
            int degree = N_v[v].size();
            for (int i = 0; i < degree; i++) {
                int c = N_v[v][i];
                float sum = llr[v];
                for (int j = 0; j < degree; j++) {
                    int c2 = N_v[v][j];
                    if (c2 != c) sum += M_cv[c2][v];
                }
                M_vc[c][v] = sum;
            }
        }

        // Check -> Variable (Min-Sum)
        for (int c = 0; c < NUM_CN; c++) {
            int degree = N_c[c].size();
            for (int i = 0; i < degree; i++) {
                int v = N_c[c][i];
                float min1 = 1e9f;
                float min2 = 1e9f;
                int idx_min = -1;
                float sign_all = 1.0f;

                for (int j = 0; j < degree; j++) {
                    int v2 = N_c[c][j];
                    float val = M_vc[c][v2];
                    sign_all *= signf(val);
                    float abs_val = fabsf(val);

                    if (abs_val < min1) {
                        min2 = min1;
                        min1 = abs_val;
                        idx_min = v2;
                    } else if (abs_val < min2) {
                        min2 = abs_val;
                    }
                }

                float s = sign_all * signf(M_vc[c][v]);
                float minval = (v == idx_min) ? min2 : min1;
                M_cv[c][v] = ALPHA * s * minval;
            }
        }
    }

    // Final decision
    for (int v = 0; v < NUM_VN; v++) {
        float sum = llr[v];
        int degree = N_v[v].size();
        for (int i = 0; i < degree; i++) {
            int c = N_v[v][i];
            sum += M_cv[c][v];
        }
        decoded[v] = (sum >= 0.0f) ? 0 : 1;
    }
}

int main() {
    vector<float> h_llr(NUM_VN);
    srand(42);
    for (int v = 0; v < NUM_VN; v++)
        h_llr[v] = 1.0f + ((float)rand()/RAND_MAX - 0.5f)*2.0f;
    h_llr[3] = -0.7f;
    h_llr[5] = -1.2f;

    vector<int> h_cpu(NUM_VN);
    vector<int> h_gpu(NUM_VN);

    cpu_ldpc_decode(h_llr, h_cpu);

    printf("Running GPU decode...\n");
    run_ldpc_decode_gpu(h_llr, h_gpu);
    printf("GPU decode finished.\n");

    printf("\nLLR input: [ ");
    for (int v = 0; v < NUM_VN; v++) printf("%.2f ", h_llr[v]);
    printf("]\n");

    printf("CPU: [ ");
    for (int v = 0; v < NUM_VN; v++) printf("%d ", h_cpu[v]);
    printf("]\n");

    printf("GPU: [ ");
    for (int v = 0; v < NUM_VN; v++) printf("%d ", h_gpu[v]);
    printf("]\n");

    int err = 0;
    for (int v = 0; v < NUM_VN; v++)
        if (h_cpu[v] != h_gpu[v]) err++;
    printf("Verification: %s (%d diffs)\n", (err == 0) ? "OK" : "FAIL", err);

    return 0;
}
