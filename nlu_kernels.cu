#include "nlu_kernels.cuh"

// ======================= Hash 3-gramas =======================
__device__ __forceinline__
uint32_t hash3(uint8_t a, uint8_t b, uint8_t c) {
    uint32_t h = 2166136261u;
    h = (h ^ a) * 16777619u;
    h = (h ^ b) * 16777619u;
    h = (h ^ c) * 16777619u;
    return h % D;
}

// ======================= Kernel: Tokenización 3-gramas =======================
__global__
void tokenize3grams(const char* __restrict__ query, int n, float* __restrict__ vq) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i + 2 >= n) return;
    uint32_t idx = hash3((uint8_t)query[i], (uint8_t)query[i+1], (uint8_t)query[i+2]);
    atomicAdd(&vq[idx], 1.0f);
}

// ======================= Kernel: Normalización L2 =======================
__global__
void l2normalize(float* __restrict__ v, int d) {
    __shared__ float ssum[256];
    float acc = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        float x = v[j];
        acc += x * x;
    }
    ssum[threadIdx.x] = acc;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset)
            ssum[threadIdx.x] += ssum[threadIdx.x + offset];
        __syncthreads();
    }

    float norm = sqrtf(ssum[0] + 1e-12f);
    __syncthreads();

    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        v[j] = v[j] / norm;
    }
}

// ======================= Kernel: Producto matriz-vector =======================
__global__
void matvecDotCos(const float* __restrict__ M, const float* __restrict__ vq,
                  float* __restrict__ scores, int K, int D) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;

    float acc = 0.f;
    for (int j = 0; j < D; ++j)
        acc += M[k * D + j] * vq[j];
    scores[k] = acc;
}

// ======================= Kernel: Decisión (argmax) =======================
__global__
void fuseDecision(const float* __restrict__ scores, int K,
                  int* __restrict__ outTop,
                  float* __restrict__ outConfidence) {
    __shared__ float sScores[128];
    __shared__ int sIndices[128];

    int tid = threadIdx.x;
    if (tid < K) {
        sScores[tid] = scores[tid];
        sIndices[tid] = tid;
    } else {
        sScores[tid] = -1e9f;
        sIndices[tid] = -1;
    }
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            if (sScores[tid + offset] > sScores[tid]) {
                sScores[tid] = sScores[tid + offset];
                sIndices[tid] = sIndices[tid + offset];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        *outTop = sIndices[0];
        *outConfidence = sScores[0];
    }
}