%%writefile main.cu
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
#include <cctype>
#include <iostream>

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
constexpr int D = 8192;
constexpr int K = 8;
constexpr int TOPK = 3;
constexpr int MAX_QUERY = 512;
constexpr int C = 4;
constexpr int N = 1<<20;
constexpr int W = 2048;

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
__global__
void tokenize3grams(const char* __restrict__ query, int n,
                    float* __restrict__ vq){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i+2 >= n) return;
    uint32_t idx = hash3((uint8_t)query[i], (uint8_t)query[i+1], (uint8_t)query[i+2]);
    atomicAdd(&vq[idx], 1.0f);
}

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

__global__
void fuseDecision(const float* __restrict__ scores, int K,
                  const float* __restrict__ meanC,
                  const float* __restrict__ maxC,
                  float consumo_umbral_alto,
                  float consumo_umbral_bajo,
                  int* __restrict__ outDecision, 
                  int* __restrict__ outTop,
                  float* __restrict__ outConfidence){
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
        int topIdx = sIndices[0];
        float topScore = sScores[0];
        
        *outTop = topIdx;
        *outConfidence = topScore;
        
        int decision = 0;
        float consumo = meanC[1];
        float temp = meanC[0];
        float ocupacion = meanC[3];
        
        if (topIdx == ENCENDER) {
            if (consumo < consumo_umbral_alto) {
                decision = 1;
            } else {
                decision = 2;
            }
        }
        else if (topIdx == APAGAR) {
            decision = 1;
        }
        else if (topIdx == CONSULTAR || topIdx == ESTADO) {
            decision = 1;
        }
        else if (topIdx == AHORRO) {
            if (consumo > consumo_umbral_bajo) {
                decision = 1;
            }
        }
        else if (topIdx == PROGRAMAR) {
            decision = 1;
        }
        else if (topIdx == DIAGNOSTICO) {
            decision = 1;
        }
        
        *outDecision = decision;
    }
}

// ======================= Host Helpers =======================
void initIntentPrototypes(std::vector<float>& M){
    srand(42);
    M.resize(K * D);
    
    for (int k=0; k<K; ++k){
        double acc=0;
        for (int j=0; j<D; ++j){
            unsigned int seed = (k+1)*1103515245u + j*12345u;
            float v = float((seed % 1000)) / 1000.0f;
            M[k*D+j] = v;
            acc += double(v)*double(v);
        }
        float n = float(std::sqrt(acc)+1e-12);
        for (int j=0; j<D; ++j) M[k*D+j] /= n;
    }
}

void synthSensors(std::vector<float>& X){
    X.resize(size_t(N)*C);
    srand(7);
    
    for (int i=0; i<N; ++i){
        float temp = 20.f + (rand()%1000)/1000.f * 10.f;
        float consumo = 500.f + (rand()%1000)/1000.f * 2000.f;
        float luz = 100.f + (rand()%1000)/1000.f * 900.f;
        float ocup = (rand()%100) < 30 ? 1.f : 0.f;
        
        X[i*C+0]=temp; 
        X[i*C+1]=consumo; 
        X[i*C+2]=luz; 
        X[i*C+3]=ocup;
    }
}

// ======================= Programación Defensiva =======================

// Sanitiza el input del usuario
bool sanitizeInput(std::string& input) {
    // 1. Verificar que no esté vacío
    if (input.empty()) {
        return false;
    }
    
    // 2. Remover espacios al inicio y final
    size_t start = input.find_first_not_of(" \t\n\r");
    size_t end = input.find_last_not_of(" \t\n\r");
    
    if (start == std::string::npos) {
        return false; // Solo espacios
    }
    
    input = input.substr(start, end - start + 1);
    
    // 3. Limitar longitud
    if (input.length() > MAX_QUERY - 1) {
        input = input.substr(0, MAX_QUERY - 1);
        printf("Input truncado a %d caracteres\n", MAX_QUERY - 1);
    }
    
    // 4. Verificar caracteres válidos (permitir español y caracteres comunes)
    bool hasValidChar = false;
    for (char c : input) {
        // Permitir letras, números, espacios y puntuación común
        if (std::isalnum(static_cast<unsigned char>(c)) || 
            std::isspace(static_cast<unsigned char>(c)) ||
            c == '.' || c == ',' || c == '?' || c == '!' || 
            c == '-' || c == '_' || c == '\'' ||
            (static_cast<unsigned char>(c) >= 128)) { // Caracteres UTF-8
            hasValidChar = true;
        }
    }
    
    if (!hasValidChar) {
        return false;
    }
    
    // 5. Convertir a minúsculas para consistencia
    std::transform(input.begin(), input.end(), input.begin(),
                   [](unsigned char c) { 
                       return (c < 128) ? std::tolower(c) : c; 
                   });
    
    return true;
}

// Lee input del usuario de forma segura
bool getUserInput(std::string& query) {
    printf("\nIngresa tu consulta (o 'salir' para terminar): ");
    
    // Limpiar buffer
    std::cin.clear();
    
    // Leer línea completa
    if (!std::getline(std::cin, query)) {
        return false; // Error de lectura o EOF
    }
    
    // Verificar comando de salida
    std::string lower_query = query;
    std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    if (lower_query == "salir" || lower_query == "exit" || 
        lower_query == "quit" || lower_query == "q") {
        return false;
    }
    
    // Sanitizar input
    if (!sanitizeInput(query)) {
        printf("Input inválido. Por favor intenta de nuevo.\n");
        return getUserInput(query); // Intentar de nuevo recursivamente
    }
    
    return true;
}

// Muestra ejemplos de comandos
void showHelp() {
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("            SMART HOME CHAT-BOX AYUDA\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    printf("Ejemplos de comandos que puedes usar:\n\n");
    printf("ENCENDER:\n");
    printf("     • enciende las luces del piso 2\n");
    printf("     • prende la calefacción\n");
    printf("     • activa el aire acondicionado\n\n");
    printf("APAGAR:\n");
    printf("     • apaga el aire acondicionado\n");
    printf("     • desactiva las luces\n");
    printf("     • apaga todo\n\n");
    printf("CONSULTAR:\n");
    printf("     • cuanto consumimos hoy en watts\n");
    printf("     • cual es la temperatura actual\n");
    printf("     • hay alguien en casa\n\n");
    printf("PROGRAMAR:\n");
    printf("     • programa la calefaccion para las 6 am\n");
    printf("     • configura las luces a las 8 pm\n\n");
    printf("ESTADO:\n");
    printf("     • cual es el estado de la casa\n");
    printf("     • muestra el resumen del sistema\n\n");
    printf("AHORRO:\n");
    printf("     • activa modo ahorro de energia\n");
    printf("     • reduce el consumo electrico\n\n");
    printf("DIAGNOSTICO:\n");
    printf("     • diagnostico del sistema electrico\n");
    printf("     • verifica el estado de los sensores\n\n");
    printf("Escribe 'ayuda' en cualquier momento para ver esto.\n");
    printf("Escribe 'salir' para terminar.\n");
    printf("═══════════════════════════════════════════════════════\n");
}

// ======================= Procesamiento de Query =======================
void processQuery(const std::string& query,
                 cudaStream_t sNLU, cudaStream_t sDATA, cudaStream_t sFUSE,
                 cudaEvent_t evStart, cudaEvent_t evStop, 
                 cudaEvent_t evNLU, cudaEvent_t evDATA,
                 char* hQ, char* dQ, float* dVQ, float* dScores, float* hScores,
                 float* dM, float* hX, float* dX,
                 float* dMean, float* dStd, float* dMax, float* dMin,
                 float* hMean, float* hStd, float* hMax, float* hMin,
                 float* dMeanHost, float* dMaxHost,
                 int* dDecision, int* dTop, float* dConfidence,
                 int& hDecision, int& hTop, float& hConfidence,
                 const char* intentNames[],
                 int queryNum) {
    
    int qn = std::min<int>(query.size(), MAX_QUERY - 1);
    
    memset(hQ, 0, MAX_QUERY); 
    memcpy(hQ, query.data(), qn);
    
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
    
    // Resultados
    printf("\n┌─────────────────────────────────────────────────────┐\n");
    printf("│ QUERY #%d                                        │\n", queryNum);
    printf("└─────────────────────────────────────────────────────┘\n");
    printf("  Input: \"%s\"\n\n", query.c_str());
    printf("Intent: %s (confianza: %.1f%%)\n", 
           intentNames[hTop], hConfidence * 100);
    printf("\nEstado de Sensores:\n");
    printf("Temperatura: %.1f°C (σ: %.2f)\n", hMean[0], hStd[0]);
    printf("Consumo: %.0fW (máx: %.0fW)\n", hMean[1], hMax[1]);
    printf("Luz: %.0f lux\n", hMean[2]);
    printf("Ocupación: %.0f%%\n\n", hMean[3]*100);
    
    const char* decisionStr = (hDecision == 1) ? "PERMITIR" : 
                              (hDecision == 2) ? "WARNING - Consumo alto" : "DENEGAR";
    const char* decisionColor = (hDecision == 1) ? "" : 
                                (hDecision == 2) ? "" : "";
    printf("Decisión: %s%s\n", decisionColor, decisionStr);
    printf("Latencia: %.3f ms\n", ms);
    printf("─────────────────────────────────────────────────────\n");
}

// ======================= Main =======================
int main(){
    printf("═══════════════════════════════════════════════════════\n");
    printf("    SMART HOME ENERGY CHAT-BOX CON CUDA\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("Inicializando sistema...\n\n");
    
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
    
    // Buffers auxiliares fuera del loop
    float *dMeanHost=nullptr, *dMaxHost=nullptr;
    CUDA_OK(cudaMalloc(&dMeanHost, C*sizeof(float)));
    CUDA_OK(cudaMalloc(&dMaxHost, C*sizeof(float)));
    
    // Nombres de intenciones
    static const char* intentNames[K] = {
        "ENCENDER", "APAGAR", "CONSULTAR", "PROGRAMAR",
        "ESTADO", "AYUDA", "AHORRO", "DIAGNOSTICO"
    };
    
    printf("Sistema inicializado correctamente!\n");
    printf("Escribe 'ayuda' para ver comandos disponibles.\n");
    printf("Escribe 'salir' para terminar.\n");
    
    // Loop principal de interacción
    int queryCount = 0;
    std::string userQuery;
    
    while (true) {
        // Obtener input del usuario
        if (!getUserInput(userQuery)) {
            printf("\nCerrando sistema...\n");
            break;
        }
        
        queryCount++;
        
        // Comando especial de ayuda
        if (userQuery == "ayuda" || userQuery == "help") {
            showHelp();
            continue;
        }
        
        // Procesar query con CUDA
        try {
            processQuery(userQuery, sNLU, sDATA, sFUSE,
                        evStart, evStop, evNLU, evDATA,
                        hQ, dQ, dVQ, dScores, hScores,
                        dM, hX, dX,
                        dMean, dStd, dMax, dMin,
                        hMean, hStd, hMax, hMin,
                        dMeanHost, dMaxHost,
                        dDecision, dTop, dConfidence,
                        hDecision, hTop, hConfidence,
                        intentNames, queryCount);
        } catch (const std::exception& e) {
            printf("Error procesando query: %s\n", e.what());
            continue;
        }
    }
    
    // Estadísticas finales
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("              ESTADÍSTICAS DE SESIÓN\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("Total de queries procesadas: %d\n", queryCount);
    printf("═══════════════════════════════════════════════════════\n");
    
    // Limpieza
    cudaFree(dMeanHost);
    cudaFree(dMaxHost);
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
