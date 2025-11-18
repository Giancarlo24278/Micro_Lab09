// CC3086 - Lab 9: Smart Home Chat-Box con IA (CUDA + CSV)
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
#include <fstream>
#include <sstream>

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
constexpr int MAX_QUERY = 512;
constexpr int C = 7; // Luces, A/C, Riego, Puerta, Ascensor, Total, Timestamp

// ======================= Estructura de Datos CSV =======================
struct SensorData {
    std::string fecha_hora;
    float timestamp;
    float luces;
    float ac;
    float riego;
    float puerta;
    float ascensor;
    float total;
};

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
void tokenize3grams(const char* __restrict__ query, int n, float* __restrict__ vq){
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

// ======================= Intenciones =======================
enum Intent { 
    CONSUMO=0, PUERTA=1, LUCES=2, AC=3, 
    ESTADO=4, AYUDA=5, ESTADISTICAS=6, COMPARAR=7 
};

// ======================= Kernel Decisión =======================
__global__
void fuseDecision(const float* __restrict__ scores, int K,
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
        *outTop = sIndices[0];
        *outConfidence = sScores[0];
    }
}

// ======================= Lectura CSV =======================
std::vector<SensorData> loadCSV(const std::string& filename) {
    std::vector<SensorData> data;
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        printf("⚠️  No se pudo abrir el archivo CSV: %s\n", filename.c_str());
        printf("    Usando datos de ejemplo...\n\n");
        return data;
    }
    
    std::string line;
    std::getline(file, line); // Skip header
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        SensorData record;
        std::string value;
        
        std::getline(ss, record.fecha_hora, ',');
        std::getline(ss, value, ','); record.timestamp = std::stof(value);
        std::getline(ss, value, ','); record.luces = std::stof(value);
        std::getline(ss, value, ','); record.ac = std::stof(value);
        std::getline(ss, value, ','); record.riego = std::stof(value);
        std::getline(ss, value, ','); record.puerta = std::stof(value);
        std::getline(ss, value, ','); record.ascensor = std::stof(value);
        std::getline(ss, value, ','); record.total = std::stof(value);
        
        data.push_back(record);
    }
    
    file.close();
    printf("✓ CSV cargado: %zu registros\n\n", data.size());
    return data;
}

// ======================= Análisis de Datos =======================
void analyzeData(const std::vector<SensorData>& data, int intent, const std::string& query) {
    if (data.empty()) {
        printf("No hay datos disponibles para analizar.\n");
        return;
    }
    
    printf("\n┌─────────────────────────────────────────────────────┐\n");
    printf("│ ANÁLISIS DE DATOS REALES                            │\n");
    printf("└─────────────────────────────────────────────────────┘\n");
    
    // Calcular estadísticas
    float total_luces = 0, total_ac = 0, total_riego = 0;
    float total_puerta = 0, total_ascensor = 0, total_consumo = 0;
    float max_puerta = 0, count_puerta = 0;
    
    for (const auto& record : data) {
        total_luces += record.luces;
        total_ac += record.ac;
        total_riego += record.riego;
        total_puerta += record.puerta;
        total_ascensor += record.ascensor;
        total_consumo += record.total;
        
        if (record.puerta > 0.1) {
            count_puerta++;
            if (record.puerta > max_puerta) max_puerta = record.puerta;
        }
    }
    
    switch(intent) {
        case PUERTA:
            printf("📊 Análisis de Puerta:\n");
            printf("   • Activaciones detectadas: %.0f eventos\n", count_puerta);
            printf("   • Consumo total: %.2f Wh\n", total_puerta);
            printf("   • Consumo máximo por evento: %.2f Wh\n", max_puerta);
            printf("   • Promedio por activación: %.2f Wh\n", 
                   count_puerta > 0 ? total_puerta/count_puerta : 0);
            break;
            
        case CONSUMO:
            printf("⚡ Consumo Total del Sistema:\n");
            printf("   • Consumo acumulado: %.2f Wh\n", total_consumo);
            printf("   • Luces: %.2f Wh (%.1f%%)\n", total_luces, 
                   (total_luces/total_consumo)*100);
            printf("   • A/C: %.2f Wh (%.1f%%)\n", total_ac, 
                   (total_ac/total_consumo)*100);
            printf("   • Riego: %.2f Wh (%.1f%%)\n", total_riego, 
                   (total_riego/total_consumo)*100);
            printf("   • Puerta: %.2f Wh (%.1f%%)\n", total_puerta, 
                   (total_puerta/total_consumo)*100);
            printf("   • Ascensor: %.2f Wh (%.1f%%)\n", total_ascensor, 
                   (total_ascensor/total_consumo)*100);
            break;
            
        case LUCES:
            printf("💡 Análisis de Iluminación:\n");
            printf("   • Consumo total: %.2f Wh\n", total_luces);
            printf("   • Porcentaje del total: %.1f%%\n", 
                   (total_luces/total_consumo)*100);
            break;
            
        case AC:
            printf("❄️  Análisis de Aire Acondicionado:\n");
            printf("   • Consumo total: %.2f Wh\n", total_ac);
            printf("   • Porcentaje del total: %.1f%%\n", 
                   (total_ac/total_consumo)*100);
            if (total_ac > total_consumo * 0.5) {
                printf("   ⚠️  El A/C representa más del 50%% del consumo\n");
            }
            break;
            
        case ESTADO:
            printf("🏠 Estado General del Sistema:\n");
            printf("   • Total de registros: %zu\n", data.size());
            printf("   • Consumo acumulado: %.2f Wh\n", total_consumo);
            printf("   • Sistema más utilizado: ");
            float max_val = std::max({total_luces, total_ac, total_riego, 
                                      total_puerta, total_ascensor});
            if (max_val == total_luces) printf("Luces");
            else if (max_val == total_ac) printf("A/C");
            else if (max_val == total_riego) printf("Riego");
            else if (max_val == total_puerta) printf("Puerta");
            else printf("Ascensor");
            printf(" (%.2f Wh)\n", max_val);
            break;
            
        case ESTADISTICAS:
            printf("📈 Estadísticas Detalladas:\n");
            printf("   • Periodo de datos: %zu registros\n", data.size());
            printf("   • Consumo promedio por registro: %.2f Wh\n", 
                   total_consumo / data.size());
            printf("\n   Desglose por dispositivo:\n");
            printf("   ├─ Luces:     %7.2f Wh (%5.1f%%)\n", total_luces, 
                   (total_luces/total_consumo)*100);
            printf("   ├─ A/C:       %7.2f Wh (%5.1f%%)\n", total_ac, 
                   (total_ac/total_consumo)*100);
            printf("   ├─ Riego:     %7.2f Wh (%5.1f%%)\n", total_riego, 
                   (total_riego/total_consumo)*100);
            printf("   ├─ Puerta:    %7.2f Wh (%5.1f%%)\n", total_puerta, 
                   (total_puerta/total_consumo)*100);
            printf("   └─ Ascensor:  %7.2f Wh (%5.1f%%)\n", total_ascensor, 
                   (total_ascensor/total_consumo)*100);
            break;
            
        default:
            printf("Consulta tu pregunta de otra manera.\n");
    }
    
    printf("─────────────────────────────────────────────────────\n");
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

// ======================= Sanitización =======================
bool sanitizeInput(std::string& input) {
    if (input.empty()) return false;
    
    size_t start = input.find_first_not_of(" \t\n\r");
    size_t end = input.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) return false;
    
    input = input.substr(start, end - start + 1);
    if (input.length() > MAX_QUERY - 1) {
        input = input.substr(0, MAX_QUERY - 1);
    }
    
    std::transform(input.begin(), input.end(), input.begin(),
                   [](unsigned char c) { 
                       return (c < 128) ? std::tolower(c) : c; 
                   });
    
    return true;
}

bool getUserInput(std::string& query) {
    printf("\n💬 Pregunta sobre tus datos: ");
    std::cin.clear();
    
    if (!std::getline(std::cin, query)) return false;
    
    std::string lower_query = query;
    std::transform(lower_query.begin(), lower_query.end(), lower_query.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    if (lower_query == "salir" || lower_query == "exit") return false;
    
    if (!sanitizeInput(query)) {
        printf("Input inválido. Intenta de nuevo.\n");
        return getUserInput(query);
    }
    
    return true;
}

void showHelp() {
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("        SMART HOME CSV ANALYZER - AYUDA\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    printf("Ejemplos de preguntas:\n\n");
    printf("🚪 PUERTA:\n");
    printf("   • cuantas veces se abrio la puerta\n");
    printf("   • consumo de la puerta\n");
    printf("   • analiza la puerta\n\n");
    printf("⚡ CONSUMO:\n");
    printf("   • cual es el consumo total\n");
    printf("   • cuanto gaste en total\n");
    printf("   • consumo de energia\n\n");
    printf("💡 LUCES:\n");
    printf("   • consumo de las luces\n");
    printf("   • cuanto gastan las luces\n\n");
    printf("❄️  AIRE ACONDICIONADO:\n");
    printf("   • consumo del aire acondicionado\n");
    printf("   • cuanto gasta el ac\n\n");
    printf("🏠 ESTADO:\n");
    printf("   • estado general\n");
    printf("   • resumen de la casa\n\n");
    printf("📊 ESTADÍSTICAS:\n");
    printf("   • dame las estadisticas\n");
    printf("   • analisis completo\n\n");
    printf("Escribe 'ayuda' para ver esto nuevamente\n");
    printf("Escribe 'salir' para terminar\n");
    printf("═══════════════════════════════════════════════════════\n");
}

// ======================= Main =======================
int main(){
    printf("═══════════════════════════════════════════════════════\n");
    printf("    SMART HOME CSV ANALYZER CON CUDA\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Cargar datos CSV
    std::string csv_path = "Consumo_Casa_inteligente - Hoja 1.csv";
    std::vector<SensorData> csvData = loadCSV(csv_path);
    
    if (csvData.empty()) {
        printf("❌ No se encontró el archivo CSV o está vacío.\n");
        printf("   Coloca el archivo en el mismo directorio que el ejecutable.\n");
        printf("   Nombre esperado: Consumo_Casa_inteligente - Hoja 1.csv\n\n");
        return 1;
    }
    
    // Inicializar CUDA
    cudaStream_t sNLU;
    CUDA_OK(cudaStreamCreate(&sNLU));
    
    cudaEvent_t evStart, evStop;
    CUDA_OK(cudaEventCreate(&evStart));
    CUDA_OK(cudaEventCreate(&evStop));
    
    std::vector<float> hM;
    initIntentPrototypes(hM);
    float *dM=nullptr;
    CUDA_OK(cudaMalloc(&dM, K*D*sizeof(float)));
    CUDA_OK(cudaMemcpy(dM, hM.data(), K*D*sizeof(float), cudaMemcpyHostToDevice));
    
    char *hQ=nullptr, *dQ=nullptr;
    float *dVQ=nullptr, *dScores=nullptr, *hScores=nullptr;
    int *dTop=nullptr;
    float *dConfidence=nullptr;
    
    CUDA_OK(cudaHostAlloc(&hQ, MAX_QUERY, cudaHostAllocDefault));
    CUDA_OK(cudaHostAlloc(&hScores, K*sizeof(float), cudaHostAllocDefault));
    CUDA_OK(cudaMalloc(&dVQ, D*sizeof(float)));
    CUDA_OK(cudaMalloc(&dScores, K*sizeof(float)));
    CUDA_OK(cudaMalloc(&dQ, MAX_QUERY));
    CUDA_OK(cudaMalloc(&dTop, sizeof(int)));
    CUDA_OK(cudaMalloc(&dConfidence, sizeof(float)));
    
    static const char* intentNames[K] = {
        "CONSUMO", "PUERTA", "LUCES", "A/C",
        "ESTADO", "AYUDA", "ESTADISTICAS", "COMPARAR"
    };
    
    printf("Sistema listo. Escribe 'ayuda' para ver comandos.\n");
    
    int queryCount = 0;
    std::string userQuery;
    
    while (true) {
        if (!getUserInput(userQuery)) {
            printf("\n👋 Cerrando sistema...\n");
            break;
        }
        
        queryCount++;
        
        if (userQuery == "ayuda" || userQuery == "help") {
            showHelp();
            continue;
        }
        
        // Procesar query con NLU
        int qn = std::min<int>(userQuery.size(), MAX_QUERY - 1);
        memset(hQ, 0, MAX_QUERY);
        memcpy(hQ, userQuery.data(), qn);
        
        CUDA_OK(cudaEventRecord(evStart, 0));
        CUDA_OK(cudaMemsetAsync(dVQ, 0, D*sizeof(float), sNLU));
        CUDA_OK(cudaMemcpyAsync(dQ, hQ, MAX_QUERY, cudaMemcpyHostToDevice, sNLU));
        
        dim3 blk(256), grd(ceilDiv(qn, (int)blk.x));
        tokenize3grams<<<grd, blk, 0, sNLU>>>(dQ, qn, dVQ);
        l2normalize<<<1, 256, 0, sNLU>>>(dVQ, D);
        
        dim3 blk2(128), grd2(ceilDiv(K, (int)blk2.x));
        matvecDotCos<<<grd2, blk2, 0, sNLU>>>(dM, dVQ, dScores, K, D);
        
        int hTop = 0;
        float hConfidence = 0.0f;
        fuseDecision<<<1, 128, 0, sNLU>>>(dScores, K, dTop, dConfidence);
        
        CUDA_OK(cudaMemcpyAsync(&hTop, dTop, sizeof(int), 
                               cudaMemcpyDeviceToHost, sNLU));
        CUDA_OK(cudaMemcpyAsync(&hConfidence, dConfidence, sizeof(float), 
                               cudaMemcpyDeviceToHost, sNLU));
        CUDA_OK(cudaStreamSynchronize(sNLU));
        CUDA_OK(cudaEventRecord(evStop, 0));
        CUDA_OK(cudaEventSynchronize(evStop));
        
        float ms=0;
        CUDA_OK(cudaEventElapsedTime(&ms, evStart, evStop));
        
        printf("\n🎯 Intent: %s (%.1f%% confianza)\n", 
               intentNames[hTop], hConfidence * 100);
        
        // Analizar datos basado en el intent
        analyzeData(csvData, hTop, userQuery);
        
        printf("\n⏱️  Tiempo de procesamiento: %.2f ms\n", ms);
    }
    
    // Limpieza
    cudaFree(dQ); cudaFree(dVQ); cudaFree(dScores); cudaFree(dM);
    cudaFree(dTop); cudaFree(dConfidence);
    cudaFreeHost(hQ); cudaFreeHost(hScores);
    cudaEventDestroy(evStart); cudaEventDestroy(evStop);
    cudaStreamDestroy(sNLU);
    
    printf("\n✓ Sesión finalizada: %d consultas procesadas\n", queryCount);
    return 0;
}