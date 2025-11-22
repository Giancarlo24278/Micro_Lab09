#include "nlu_engine.h"
#include "nlu_kernels.cuh"
#include <cmath>
#include <cstring>
#include <algorithm>

NLUEngine::NLUEngine() : dM(nullptr), hQ(nullptr), dQ(nullptr), 
                         dVQ(nullptr), dScores(nullptr), hScores(nullptr),
                         dTop(nullptr), dConfidence(nullptr) {
}

NLUEngine::~NLUEngine() {
    cleanup();
}

void NLUEngine::initialize() {
    // Crear stream y eventos
    CUDA_OK(cudaStreamCreate(&stream));
    CUDA_OK(cudaEventCreate(&evStart));
    CUDA_OK(cudaEventCreate(&evStop));
    
    // Inicializar prototipos
    initIntentPrototypes();
    
    // Allocar memoria device
    CUDA_OK(cudaMalloc(&dM, K * D * sizeof(float)));
    CUDA_OK(cudaMemcpy(dM, hM.data(), K * D * sizeof(float), 
                       cudaMemcpyHostToDevice));
    
    // Allocar memoria para query y resultados
    CUDA_OK(cudaHostAlloc(&hQ, MAX_QUERY, cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc(&hScores, K * sizeof(float), cudaHostAllocDefault));
    CUDA_OK(cudaMalloc(&dVQ, D * sizeof(float)));
    CUDA_OK(cudaMalloc(&dScores, K * sizeof(float)));
    CUDA_OK(cudaMalloc(&dQ, MAX_QUERY));
    CUDA_OK(cudaMalloc(&dTop, sizeof(int)));
    CUDA_OK(cudaMalloc(&dConfidence, sizeof(float)));
}

void NLUEngine::initIntentPrototypes() {
    srand(42);
    hM.resize(K * D);
    
    for (int k = 0; k < K; ++k) {
        double acc = 0;
        for (int j = 0; j < D; ++j) {
            unsigned int seed = (k + 1) * 1103515245u + j * 12345u;
            float v = float((seed % 1000)) / 1000.0f;
            hM[k * D + j] = v;
            acc += double(v) * double(v);
        }
        float n = float(std::sqrt(acc) + 1e-12);
        for (int j = 0; j < D; ++j) 
            hM[k * D + j] /= n;
    }
}

void NLUEngine::processQuery(const std::string& query, int& topIntent, 
                              float& confidence, float& processingTime) {
    // Copiar query
    int qn = std::min<int>(query.size(), MAX_QUERY - 1);
    memset(hQ, 0, MAX_QUERY);
    memcpy(hQ, query.data(), qn);
    
    // Iniciar timer
    CUDA_OK(cudaEventRecord(evStart, 0));
    
    // Procesar query
    CUDA_OK(cudaMemsetAsync(dVQ, 0, D * sizeof(float), stream));
    CUDA_OK(cudaMemcpyAsync(dQ, hQ, MAX_QUERY, cudaMemcpyHostToDevice, stream));
    
    // Tokenizar 3-gramas
    dim3 blk(256), grd(ceilDiv(qn, (int)blk.x));
    tokenize3grams<<<grd, blk, 0, stream>>>(dQ, qn, dVQ);
    
    // Normalizar
    l2normalize<<<1, 256, 0, stream>>>(dVQ, D);
    
    // Calcular scores
    dim3 blk2(128), grd2(ceilDiv(K, (int)blk2.x));
    matvecDotCos<<<grd2, blk2, 0, stream>>>(dM, dVQ, dScores, K, D);
    
    // Obtener decisión
    fuseDecision<<<1, 128, 0, stream>>>(dScores, K, dTop, dConfidence);
    
    // Copiar resultados
    CUDA_OK(cudaMemcpyAsync(&topIntent, dTop, sizeof(int),
                           cudaMemcpyDeviceToHost, stream));
    CUDA_OK(cudaMemcpyAsync(&confidence, dConfidence, sizeof(float),
                           cudaMemcpyDeviceToHost, stream));
    CUDA_OK(cudaStreamSynchronize(stream));
    
    // Parar timer
    CUDA_OK(cudaEventRecord(evStop, 0));
    CUDA_OK(cudaEventSynchronize(evStop));
    CUDA_OK(cudaEventElapsedTime(&processingTime, evStart, evStop));
}

void NLUEngine::cleanup() {
    if (dQ) cudaFree(dQ);
    if (dVQ) cudaFree(dVQ);
    if (dScores) cudaFree(dScores);
    if (dM) cudaFree(dM);
    if (dTop) cudaFree(dTop);
    if (dConfidence) cudaFree(dConfidence);
    if (hQ) cudaFreeHost(hQ);
    if (hScores) cudaFreeHost(hScores);
    
    cudaEventDestroy(evStart);
    cudaEventDestroy(evStop);
    cudaStreamDestroy(stream);
}