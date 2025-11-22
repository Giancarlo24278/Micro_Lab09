#ifndef NLU_KERNELS_CUH
#define NLU_KERNELS_CUH

#include "common.h"

// ======================= Declaración de Kernels =======================
__global__ void tokenize3grams(const char* __restrict__ query, int n, 
                               float* __restrict__ vq);

__global__ void l2normalize(float* __restrict__ v, int d);

__global__ void matvecDotCos(const float* __restrict__ M, 
                             const float* __restrict__ vq,
                             float* __restrict__ scores, int K, int D);

__global__ void fuseDecision(const float* __restrict__ scores, int K,
                            int* __restrict__ outTop,
                            float* __restrict__ outConfidence);

#endif // NLU_KERNELS_CUH