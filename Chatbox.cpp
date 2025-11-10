// CC3086 - Lab 9: Smart Home Chat-Box con IA (CUDA)
// Dominio: Smart Home / Energy Management
// Compilar: nvcc -O3 -std=c++17 main.cu -o chatbox_cuda

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>

// ======================= Utilidades =======================
#define CUDA_OK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line){
    if (code != cudaSuccess){ 
        fprintf(stderr,"CUDA Error: %s %s %d\n",
        cudaGetErrorString(code), file, line); 
        exit(code); 
    }
}

inline int ceilDiv(int a, int b){ return (a + b - 1) / b; }

// ======================= Parámetros =======================
// Vectorización 3-gramas -> espacio D
constexpr int D = 8192;           // dimensión de representación
constexpr int K = 8;              // intenciones: encender, apagar, consultar, programar, estado, ayuda, ahorro, diagnostico
constexpr int TOPK = 3;           // sugerencias
constexpr int MAX_QUERY = 512;    // longitud máx. de consulta

// Sensores: temperatura, consumo_w, luz_lux, ocupacion
constexpr int C = 4;              
constexpr int N = 1<<20;          // ~1M muestras
constexpr int W = 2048;           // ventana para stats

// ======================= Hash 3-gramas =======================
__device__ __forceinline__
uint32_t hash3(uint8_t a, uint8_t b, uint8_t c){
    uint32_t h = 2166136261u;
    h = (h ^ a) * 16777619u;
    h = (h ^ b) * 16777619u;
    h = (h ^ c) * 16777619u;
    return h % D;
}

// ======================= Kernels NLU =======================
// Kernel 1: Tokenización 3-gramas
__global__
void tokenize3grams(const char* __restrict__ query, int n,
                    float* __restrict__ vq){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i+2 >= n) return;
    uint32_t idx = hash3((uint8_t)query[i], (uint8_t)query[i+1], (uint8_t)query[i+2]);
    atomicAdd(&vq[idx], 1.0f);
}

// Kernel 2: Normalización L2
__global__
void l2normalize(float* __restrict__ v, int d){
    __shared__ float ssum[256];
    float acc = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x){
        float x = v[j];
        acc += x*x;
    }
    ssum[threadIdx.x] = acc;
    __syncthreads();
    
    for (int offset = blockDim.x>>1; offset > 0; offset >>= 1){
        if (threadIdx.x < offset) 
            ssum[threadIdx.x] += ssum[threadIdx.x+offset];
        __syncthreads();
    }
    
    float norm = sqrtf(ssum[0] + 1e-12f);
    __syncthreads();
    
    for (int j = threadIdx.x; j < d; j += blockDim.x){
        v[j] = v[j] / norm;
    }
}

// Kernel 3: Similitud coseno (scores = M·vq)
__global__
void matvecDotCos(const float* __restrict__ M, const float* __restrict__ vq,
                  float* __restrict__ scores, int K, int D){
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    
    float acc = 0.f;
    for (int j = 0; j < D; ++j) 
        acc += M[k*D + j] * vq[j];
    scores[k] = acc;
}

// ======================= Kernels Sensores =======================
// Kernel 4: Estadísticas de ventana (mean, std, max, min)
__global__
void window_stats_advanced(const float* __restrict__ X, int N, int C, int W,
                          float* __restrict__ mean_out, 
                          float* __restrict__ std_out,
                          float* __restrict__ max_out,
                          float* __restrict__ min_out){
    int c = blockIdx.x;
    if (c >= C) return;
    
    float sum = 0.f, sum2 = 0.f;
    float local_max = -1e9f, local_min = 1e9f;
    int start = max(0, N - W);
    
    for (int i = threadIdx.x; i < W; i += blockDim.x){
        float v = X[(start + i)*C + c];
        sum += v;
        sum2 += v*v;
        local_max = fmaxf(local_max, v);
        local_min = fminf(local_min, v);
    }
    
    __shared__ float ssum[256], ssum2[256], smax[256], smin[256];
    ssum[threadIdx.x] = sum;
    ssum2[threadIdx.x] = sum2;
    smax[threadIdx.x] = local_max;
    smin[threadIdx.x] = local_min;
    __syncthreads();
    
    for (int off = blockDim.x>>1; off > 0; off >>= 1){
        if (threadIdx.x < off){
            ssum[threadIdx.x] += ssum[threadIdx.x+off];
            ssum2[threadIdx.x] += ssum2[threadIdx.x+off];
            smax[threadIdx.x] = fmaxf(smax[threadIdx.x], smax[threadIdx.x+off]);
            smin[threadIdx.x] = fminf(smin[threadIdx.x], smin[threadIdx.x+off]);
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0){
        float m = ssum[0] / W;
        float var = fmaxf(ssum2[0]/W - m*m, 0.f);
        mean_out[c] = m;
        std_out[c] = sqrtf(var);
        max_out[c] = smax[0];
        min_out[c] = smin[0];
    }
}

// ======================= Kernel Fusión/Decisión =======================
enum Intent { 
    ENCENDER=0, APAGAR=1, CONSULTAR=2, PROGRAMAR=3, 
    ESTADO=4, AYUDA=5, AHORRO=6, DIAGNOSTICO=7 
};

// Kernel 5: Fusión y decisión
__global__
void fuseDecision(const float* __restrict__ scores, int K,
                  const float* __restrict__ meanC,
                  const float* __restrict__ maxC,
                  float consumo_umbral_alto,
                  float consumo_umbral_bajo,
                  int* __restrict__ outDecision, 
                  int* __restrict__ outTop,
                  float* __restrict__ outConfidence){
    __shared__ int topIdx;
    __shared__ float topScore;
    
    if (threadIdx.x == 0){ 
        topIdx = 0; 
        topScore = scores[0]; 
    }
    __syncthreads();
    
    // Encontrar top intent
    for (int k = threadIdx.x; k < K; k += blockDim.x){
        float s = scores[k];
        if (s > topScore){ 
            topScore = s; 
            topIdx = k; 
        }
    }
    __syncthreads();
    
    if (threadIdx.x == 0){
        *outTop = topIdx;
        *outConfidence = topScore;
        
        int decision = 0; // 0=denegar, 1=permitir, 2=warning
        float consumo = meanC[1]; // consumo_w
        float temp = meanC[0];
        float ocupacion = meanC[3];
        
        // Lógica de decisión por intención
        if (topIdx == ENCENDER) {
            // Permitir encender si consumo no está muy alto
            if (consumo < consumo_umbral_alto) {
                decision = 1;
            } else {
                decision = 2; // warning: consumo alto
            }
        }
        else if (topIdx == APAGAR) {
            // Siempre permitir apagar
            decision = 1;
        }
        else if (topIdx == CONSULTAR || topIdx == ESTADO) {
            // Siempre permitir consultas
            decision = 1;
        }
        else if (topIdx == AHORRO) {
            // Modo ahorro: solo si consumo está alto
            if (consumo > consumo_umbral_bajo) {
                decision = 1;
            }
        }
        else if (topIdx == PROGRAMAR) {
            // Permitir programación
            decision = 1;
        }
        else if (topIdx == DIAGNOSTICO) {
            // Diagnóstico siempre disponible
            decision = 1;
        }
        
        *outDecision = decision;
    }
}

// ======================= Host Helpers =======================
void initIntentPrototypes(std::vector<float>& M){
    // KxD matriz de prototipos por intención
    srand(42);
    M.resize(K * D);
    
    // Patrones diferentes por cada intención
    for (int k=0; k<K; ++k){
        double acc=0;
        for (int j=0; j<D; ++j){
            // Semilla única por intención
            unsigned int seed = (k+1)*1103515245u + j*12345u;
            float v = float((seed % 1000)) / 1000.0f;
            M[k*D+j] = v;
            acc += double(v)*double(v);
        }
        float n = float(std::sqrt(acc)+1e-12);
        for (int j=0; j<D; ++j) M[k*D+j] /= n;
    }
}

std::vector<std::string> getDemoQueries(){
    return {
        "enciende las luces del piso 2",
        "apaga el aire acondicionado",
        "cuanto consumimos hoy en watts",
        "programa la calefaccion para las 6 am",
        "cual es el estado de la casa",
        "ayuda con los comandos disponibles",
        "activa modo ahorro de energia",
        "diagnostico del sistema electrico"
    };
}

void synthSensors(std::vector<float>& X){
    // N x C: {temperatura, consumo_w, luz_lux, ocupacion}
    X.resize(size_t(N)*C);
    srand(7);
    
    for (int i=0; i<N; ++i){
        float temp = 20.f + (rand()%1000)/1000.f * 10.f;      // 20-30°C
        float consumo = 500.f + (rand()%1000)/1000.f * 2000.f; // 500-2500W
        float luz = 100.f + (rand()%1000)/1000.f * 900.f;      // 100-1000 lux
        float ocup = (rand()%100) < 30 ? 1.f : 0.f;            // 30% ocupado
        
        X[i*C+0]=temp; 
        X[i*C+1]=consumo; 
        X[i*C+2]=luz; 
        X[i*C+3]=ocup;
    }
}

// ======================= Main =======================
int main(){
    printf("=== Smart Home Energy Chat-Box con CUDA ===\n\n");
    
    // Streams
    cudaStream_t sNLU, sDATA, sFUSE, sLOG;
    CUDA_OK(cudaStreamCreate(&sNLU));
    CUDA_OK(cudaStreamCreate(&sDATA));
    CUDA_OK(cudaStreamCreate(&sFUSE));
    CUDA_OK(cudaStreamCreate(&sLOG));
    
    // Eventos
    cudaEvent_t evStart, evStop, evNLU, evDATA;
    CUDA_OK(cudaEventCreate(&evStart));
    CUDA_OK(cudaEventCreate(&evStop));
    CUDA_OK(cudaEventCreate(&evNLU));
    CUDA_OK(cudaEventCreate(&evDATA));
    
    // Intent prototypes M(KxD)
    std::vector<float> hM; 
    initIntentPrototypes(hM);
    float *dM=nullptr;
    CUDA_OK(cudaMalloc(&dM, K*D*sizeof(float)));
    CUDA_OK(cudaMemcpy(dM, hM.data(), K*D*sizeof(float), cudaMemcpyHostToDevice));
    
    // Buffers NLU (pinned)
    char *hQ=nullptr;
    float *hVQ=nullptr, *dVQ=nullptr, *dScores=nullptr, *hScores=nullptr;
    CUDA_OK(cudaHostAlloc(&hQ, MAX_QUERY, cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc(&hVQ, D*sizeof(float), cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc(&hScores, K*sizeof(float), cudaHostAllocDefault));
    CUDA_OK(cudaMalloc(&dVQ, D*sizeof(float)));
    CUDA_OK(cudaMalloc(&dScores, K*sizeof(float)));
    
    // Sensores (pinned)
    std::vector<float> hXvec; 
    synthSensors(hXvec);
    float *hX=nullptr;
    CUDA_OK(cudaHostAlloc(&hX, size_t(N)*C*sizeof(float), cudaHostAllocDefault));
    memcpy(hX, hXvec.data(), size_t(N)*C*sizeof(float));
    
    float *dX=nullptr, *dMean=nullptr, *dStd=nullptr, *dMax=nullptr, *dMin=nullptr;
    float hMean[C]={0}, hStd[C]={0}, hMax[C]={0}, hMin[C]={0};
    CUDA_OK(cudaMalloc(&dX, size_t(N)*C*sizeof(float)));
    CUDA_OK(cudaMalloc(&dMean, C*sizeof(float)));
    CUDA_OK(cudaMalloc(&dStd, C*sizeof(float)));
    CUDA_OK(cudaMalloc(&dMax, C*sizeof(float)));
    CUDA_OK(cudaMalloc(&dMin, C*sizeof(float)));
    
    // Fusión
    int *dDecision=nullptr, *dTop=nullptr;
    float *dConfidence=nullptr;
    int hDecision=0, hTop=-1;
    float hConfidence=0.f;
    CUDA_OK(cudaMalloc(&dDecision, sizeof(int)));
    CUDA_OK(cudaMalloc(&dTop, sizeof(int)));
    CUDA_OK(cudaMalloc(&dConfidence, sizeof(float)));
    
    char *dQ=nullptr;
    CUDA_OK(cudaMalloc(&dQ, MAX_QUERY));
    
    // Nombres de intenciones
    static const char* intentNames[K] = {
        "ENCENDER", "APAGAR", "CONSULTAR", "PROGRAMAR",
        "ESTADO", "AYUDA", "AHORRO", "DIAGNOSTICO"
    };
    
    // Procesar múltiples queries
    auto queries = getDemoQueries();
    std::vector<float> latencies;
    
    printf("Procesando %zu consultas...\n\n", queries.size());
    
    for (size_t qi = 0; qi < queries.size(); ++qi) {
        std::string q = queries[qi];
        int qn = std::min<int>(q.size(), MAX_QUERY);
        
        memset(hQ, 0, MAX_QUERY); 
        memcpy(hQ, q.data(), qn);
        
        // Pipeline asíncrono
        CUDA_OK(cudaEventRecord(evStart, 0));
        
        // === STREAM NLU ===
        CUDA_OK(cudaMemsetAsync(dVQ, 0, D*sizeof(float), sNLU));
        CUDA_OK(cudaMemcpyAsync(dQ, hQ, MAX_QUERY, cudaMemcpyHostToDevice, sNLU));
        
        dim3 blk(256), grd(ceilDiv(qn, (int)blk.x));
        tokenize3grams<<<grd, blk, 0, sNLU>>>(dQ, qn, dVQ);
        l2normalize<<<1, 256, 0, sNLU>>>(dVQ, D);
        
        dim3 blk2(128), grd2(ceilDiv(K, (int)blk2.x));
        matvecDotCos<<<grd2, blk2, 0, sNLU>>>(dM, dVQ, dScores, K, D);
        CUDA_OK(cudaMemcpyAsync(hScores, dScores, K*sizeof(float), 
                               cudaMemcpyDeviceToHost, sNLU));
        CUDA_OK(cudaEventRecord(evNLU, sNLU));
        
        // === STREAM DATA ===
        CUDA_OK(cudaMemcpyAsync(dX, hX, size_t(N)*C*sizeof(float), 
                               cudaMemcpyHostToDevice, sDATA));
        window_stats_advanced<<<C, 256, 0, sDATA>>>(dX, N, C, W, dMean, dStd, dMax, dMin);
        CUDA_OK(cudaMemcpyAsync(hMean, dMean, C*sizeof(float), 
                               cudaMemcpyDeviceToHost, sDATA));
        CUDA_OK(cudaMemcpyAsync(hStd, dStd, C*sizeof(float), 
                               cudaMemcpyDeviceToHost, sDATA));
        CUDA_OK(cudaMemcpyAsync(hMax, dMax, C*sizeof(float), 
                               cudaMemcpyDeviceToHost, sDATA));
        CUDA_OK(cudaMemcpyAsync(hMin, dMin, C*sizeof(float), 
                               cudaMemcpyDeviceToHost, sDATA));
        CUDA_OK(cudaEventRecord(evDATA, sDATA));
        
        // Esperar ambos streams
        CUDA_OK(cudaStreamWaitEvent(sFUSE, evNLU, 0));
        CUDA_OK(cudaStreamWaitEvent(sFUSE, evDATA, 0));
        
        // === STREAM FUSE ===
        float *dMeanHost=nullptr;
        float *dMaxHost=nullptr;
        CUDA_OK(cudaMalloc(&dMeanHost, C*sizeof(float)));
        CUDA_OK(cudaMalloc(&dMaxHost, C*sizeof(float)));
        CUDA_OK(cudaMemcpyAsync(dMeanHost, hMean, C*sizeof(float), 
                               cudaMemcpyHostToDevice, sFUSE));
        CUDA_OK(cudaMemcpyAsync(dMaxHost, hMax, C*sizeof(float), 
                               cudaMemcpyHostToDevice, sFUSE));
        
        fuseDecision<<<1, 128, 0, sFUSE>>>(dScores, K, dMeanHost, dMaxHost,
                                           2000.f, 1000.f, dDecision, dTop, dConfidence);
        CUDA_OK(cudaMemcpyAsync(&hDecision, dDecision, sizeof(int), 
                               cudaMemcpyDeviceToHost, sFUSE));
        CUDA_OK(cudaMemcpyAsync(&hTop, dTop, sizeof(int), 
                               cudaMemcpyDeviceToHost, sFUSE));
        CUDA_OK(cudaMemcpyAsync(&hConfidence, dConfidence, sizeof(float), 
                               cudaMemcpyDeviceToHost, sFUSE));
        
        CUDA_OK(cudaStreamSynchronize(sFUSE));
        CUDA_OK(cudaEventRecord(evStop, 0));
        CUDA_OK(cudaEventSynchronize(evStop));
        
        float ms=0; 
        CUDA_OK(cudaEventElapsedTime(&ms, evStart, evStop));
        latencies.push_back(ms);
        
        // Resultados
        printf("=== Query %zu ===\n", qi+1);
        printf("Input: \"%s\"\n", q.c_str());
        printf("Intent: %s (confidence: %.3f)\n", intentNames[hTop], hConfidence);
        printf("Sensor Stats:\n");
        printf("  Temperatura: %.1f°C (std: %.2f)\n", hMean[0], hStd[0]);
        printf("  Consumo: %.0fW (max: %.0fW)\n", hMean[1], hMax[1]);
        printf("  Luz: %.0f lux\n", hMean[2]);
        printf("  Ocupación: %.0f%%\n", hMean[3]*100);
        
        const char* decisionStr = (hDecision == 1) ? "✓ PERMITIR" : 
                                  (hDecision == 2) ? "⚠ WARNING" : "✗ DENEGAR";
        printf("Decision: %s\n", decisionStr);
        printf("Latencia: %.3f ms\n\n", ms);
        
        cudaFree(dMeanHost);
        cudaFree(dMaxHost);
    }
    
    // Estadísticas finales
    float avg_latency = 0.f, min_lat = 1e9f, max_lat = 0.f;
    for (float lat : latencies) {
        avg_latency += lat;
        min_lat = std::min(min_lat, lat);
        max_lat = std::max(max_lat, lat);
    }
    avg_latency /= latencies.size();
    
    printf("=== Métricas de Rendimiento ===\n");
    printf("Queries procesadas: %zu\n", queries.size());
    printf("Latencia promedio: %.3f ms\n", avg_latency);
    printf("Latencia min: %.3f ms\n", min_lat);
    printf("Latencia max: %.3f ms\n", max_lat);
    printf("QPS estimado: %.1f queries/sec\n", 1000.0f / avg_latency);
    
    // Limpieza
    cudaFree(dQ); cudaFree(dVQ); cudaFree(dScores); cudaFree(dM);
    cudaFree(dX); cudaFree(dMean); cudaFree(dStd); cudaFree(dMax); cudaFree(dMin);
    cudaFree(dDecision); cudaFree(dTop); cudaFree(dConfidence);
    cudaFreeHost(hQ); cudaFreeHost(hVQ); cudaFreeHost(hScores); cudaFreeHost(hX);
    cudaEventDestroy(evStart); cudaEventDestroy(evStop);
    cudaEventDestroy(evNLU); cudaEventDestroy(evDATA);
    cudaStreamDestroy(sNLU); cudaStreamDestroy(sDATA); 
    cudaStreamDestroy(sFUSE); cudaStreamDestroy(sLOG);
    
    printf("\n=== Finalizado exitosamente ===\n");
    return 0;
}