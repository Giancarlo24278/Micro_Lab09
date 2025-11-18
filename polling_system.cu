// polling_system.cu
// Sistema de Polling para Google Sheets con CUDA
// Consulta datos cada 3-5 segundos y procesa con GPU

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <curl/curl.h>
#include <json/json.h>

// ======================= Configuración =======================
const char* SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/1d3RnoKpjoudoDzbbDu_d0ED5AwQKYIWqx271EEZSNaU/export?format=csv";
constexpr int POLLING_INTERVAL_SEC = 3;
constexpr int MAX_RECORDS = 10000;

// ======================= Estructura de Datos =======================
struct SensorRecord {
    char fecha_hora[32];
    float timestamp;
    float luces;
    float ac;
    float riego;
    float puerta;
    float ascensor;
    float total;
};

// ======================= Callback para CURL =======================
struct MemoryStruct {
    char *memory;
    size_t size;
};

static size_t WriteMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    size_t realsize = size * nmemb;
    struct MemoryStruct *mem = (struct MemoryStruct *)userp;
    
    char *ptr = (char*)realloc(mem->memory, mem->size + realsize + 1);
    if(!ptr) {
        printf("Error: no hay suficiente memoria\n");
        return 0;
    }
    
    mem->memory = ptr;
    memcpy(&(mem->memory[mem->size]), contents, realsize);
    mem->size += realsize;
    mem->memory[mem->size] = 0;
    
    return realsize;
}

// ======================= Kernels de Análisis =======================
__global__
void computeStatistics(const SensorRecord* __restrict__ data, int n,
                      float* __restrict__ totals,
                      float* __restrict__ maxs,
                      int* __restrict__ counts) {
    // totals: [luces, ac, riego, puerta, ascensor, total_consumo]
    // maxs: mismo orden
    // counts: [total_registros, puerta_activaciones]
    
    __shared__ float s_totals[6];
    __shared__ float s_maxs[6];
    __shared__ int s_counts[2];
    
    if (threadIdx.x < 6) {
        s_totals[threadIdx.x] = 0.0f;
        s_maxs[threadIdx.x] = 0.0f;
    }
    if (threadIdx.x < 2) {
        s_counts[threadIdx.x] = 0;
    }
    __syncthreads();
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        const SensorRecord& r = data[idx];
        
        atomicAdd(&s_totals[0], r.luces);
        atomicAdd(&s_totals[1], r.ac);
        atomicAdd(&s_totals[2], r.riego);
        atomicAdd(&s_totals[3], r.puerta);
        atomicAdd(&s_totals[4], r.ascensor);
        atomicAdd(&s_totals[5], r.total);
        
        atomicMax((int*)&s_maxs[0], __float_as_int(r.luces));
        atomicMax((int*)&s_maxs[1], __float_as_int(r.ac));
        atomicMax((int*)&s_maxs[2], __float_as_int(r.riego));
        atomicMax((int*)&s_maxs[3], __float_as_int(r.puerta));
        atomicMax((int*)&s_maxs[4], __float_as_int(r.ascensor));
        atomicMax((int*)&s_maxs[5], __float_as_int(r.total));
        
        if (r.puerta > 0.1f) {
            atomicAdd(&s_counts[1], 1);
        }
        
        atomicAdd(&s_counts[0], 1);
    }
    __syncthreads();
    
    if (threadIdx.x < 6) {
        atomicAdd(&totals[threadIdx.x], s_totals[threadIdx.x]);
        atomicMax((int*)&maxs[threadIdx.x], __float_as_int(s_maxs[threadIdx.x]));
    }
    if (threadIdx.x < 2) {
        atomicAdd(&counts[threadIdx.x], s_counts[threadIdx.x]);
    }
}

__global__
void detectAnomalies(const SensorRecord* __restrict__ data, int n,
                    float threshold, int* __restrict__ anomaly_indices,
                    int* __restrict__ anomaly_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        const SensorRecord& r = data[idx];
        
        // Detectar consumos anormalmente altos
        if (r.total > threshold) {
            int pos = atomicAdd(anomaly_count, 1);
            if (pos < 100) { // Guardar hasta 100 anomalías
                anomaly_indices[pos] = idx;
            }
        }
    }
}

// ======================= Parser CSV =======================
class CSVParser {
public:
    static std::vector<SensorRecord> parse(const char* csv_data) {
        std::vector<SensorRecord> records;
        std::string data(csv_data);
        
        size_t pos = 0;
        // Skip header
        pos = data.find('\n', pos);
        if (pos == std::string::npos) return records;
        pos++;
        
        while (pos < data.size()) {
            size_t line_end = data.find('\n', pos);
            if (line_end == std::string::npos) line_end = data.size();
            
            std::string line = data.substr(pos, line_end - pos);
            pos = line_end + 1;
            
            if (line.empty()) continue;
            
            SensorRecord record;
            if (parseLine(line, record)) {
                records.push_back(record);
            }
        }
        
        return records;
    }
    
private:
    static bool parseLine(const std::string& line, SensorRecord& record) {
        std::vector<std::string> fields;
        size_t start = 0;
        
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == ',') {
                fields.push_back(line.substr(start, i - start));
                start = i + 1;
            }
        }
        fields.push_back(line.substr(start));
        
        if (fields.size() < 8) return false;
        
        try {
            strncpy(record.fecha_hora, fields[0].c_str(), 31);
            record.fecha_hora[31] = '\0';
            record.timestamp = std::stof(fields[1]);
            record.luces = std::stof(fields[2]);
            record.ac = std::stof(fields[3]);
            record.riego = std::stof(fields[4]);
            record.puerta = std::stof(fields[5]);
            record.ascensor = std::stof(fields[6]);
            record.total = std::stof(fields[7]);
            return true;
        } catch (...) {
            return false;
        }
    }
};

// ======================= Sistema de Polling =======================
class PollingSystem {
private:
    CURL *curl;
    CURLcode res;
    struct MemoryStruct chunk;
    
    SensorRecord *d_data;
    float *d_totals, *d_maxs;
    int *d_counts, *d_anomaly_indices, *d_anomaly_count;
    
    cudaStream_t stream;
    cudaEvent_t evStart, evStop;
    
    std::vector<SensorRecord> h_data;
    float h_totals[6], h_maxs[6];
    int h_counts[2];
    
public:
    PollingSystem() {
        // Inicializar CURL
        curl_global_init(CURL_GLOBAL_ALL);
        curl = curl_easy_init();
        
        chunk.memory = (char*)malloc(1);
        chunk.size = 0;
        
        // Inicializar CUDA
        cudaStreamCreate(&stream);
        cudaEventCreate(&evStart);
        cudaEventCreate(&evStop);
        
        cudaMalloc(&d_data, MAX_RECORDS * sizeof(SensorRecord));
        cudaMalloc(&d_totals, 6 * sizeof(float));
        cudaMalloc(&d_maxs, 6 * sizeof(float));
        cudaMalloc(&d_counts, 2 * sizeof(int));
        cudaMalloc(&d_anomaly_indices, 100 * sizeof(int));
        cudaMalloc(&d_anomaly_count, sizeof(int));
        
        printf("✓ Sistema de polling inicializado\n");
        printf("  URL: %s\n", SHEET_CSV_URL);
        printf("  Intervalo: %d segundos\n\n", POLLING_INTERVAL_SEC);
    }
    
    ~PollingSystem() {
        cudaFree(d_data);
        cudaFree(d_totals);
        cudaFree(d_maxs);
        cudaFree(d_counts);
        cudaFree(d_anomaly_indices);
        cudaFree(d_anomaly_count);
        
        cudaEventDestroy(evStart);
        cudaEventDestroy(evStop);
        cudaStreamDestroy(stream);
        
        curl_easy_cleanup(curl);
        curl_global_cleanup();
        free(chunk.memory);
    }
    
    bool fetchData() {
        chunk.size = 0;
        
        if (!curl) return false;
        
        curl_easy_setopt(curl, CURLOPT_URL, SHEET_CSV_URL);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteMemoryCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
        
        res = curl_easy_perform(curl);
        
        if (res != CURLE_OK) {
            fprintf(stderr, "❌ Error en descarga: %s\n", curl_easy_strerror(res));
            return false;
        }
        
        return true;
    }
    
    void processData() {
        if (chunk.size == 0) return;
        
        cudaEventRecord(evStart, stream);
        
        // Parse CSV
        h_data = CSVParser::parse(chunk.memory);
        int n = std::min((int)h_data.size(), MAX_RECORDS);
        
        if (n == 0) {
            printf("⚠️  No se encontraron datos válidos\n");
            return;
        }
        
        // Copiar a GPU
        cudaMemcpyAsync(d_data, h_data.data(), n * sizeof(SensorRecord),
                       cudaMemcpyHostToDevice, stream);
        
        // Reset estadísticas
        cudaMemsetAsync(d_totals, 0, 6 * sizeof(float), stream);
        cudaMemsetAsync(d_maxs, 0, 6 * sizeof(float), stream);
        cudaMemsetAsync(d_counts, 0, 2 * sizeof(int), stream);
        cudaMemsetAsync(d_anomaly_count, 0, sizeof(int), stream);
        
        // Ejecutar kernels
        dim3 block(256);
        dim3 grid((n + block.x - 1) / block.x);
        
        computeStatistics<<<grid, block, 0, stream>>>(d_data, n, d_totals, d_maxs, d_counts);
        
        // Detectar anomalías (consumo > 50 Wh)
        detectAnomalies<<<grid, block, 0, stream>>>(d_data, n, 50.0f, 
                                                     d_anomaly_indices, d_anomaly_count);
        
        // Copiar resultados
        cudaMemcpyAsync(h_totals, d_totals, 6 * sizeof(float), 
                       cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_maxs, d_maxs, 6 * sizeof(float), 
                       cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(h_counts, d_counts, 2 * sizeof(int), 
                       cudaMemcpyDeviceToHost, stream);
        
        cudaStreamSynchronize(stream);
        cudaEventRecord(evStop, stream);
        cudaEventSynchronize(evStop);
        
        float ms = 0;
        cudaEventElapsedTime(&ms, evStart, evStop);
        
        // Mostrar resultados
        displayResults(n, ms);
    }
    
    void displayResults(int n, float processing_time) {
        auto now = std::chrono::system_clock::now();
        std::time_t now_c = std::chrono::system_clock::to_time_t(now);
        
        printf("\n╔═══════════════════════════════════════════════════════╗\n");
        printf("║  📊 ACTUALIZACIÓN DE DATOS - %s", std::ctime(&now_c));
        printf("╠═══════════════════════════════════════════════════════╣\n");
        printf("║  Registros procesados: %-6d                        ║\n", n);
        printf("║  Tiempo de procesamiento: %.2f ms                    ║\n", processing_time);
        printf("╠═══════════════════════════════════════════════════════╣\n");
        printf("║  CONSUMO ACUMULADO:                                   ║\n");
        printf("║  ├─ Total:     %10.2f Wh                        ║\n", h_totals[5]);
        printf("║  ├─ Luces:     %10.2f Wh (%5.1f%%)              ║\n", 
               h_totals[0], (h_totals[0]/h_totals[5])*100);
        printf("║  ├─ A/C:       %10.2f Wh (%5.1f%%)              ║\n", 
               h_totals[1], (h_totals[1]/h_totals[5])*100);
        printf("║  ├─ Riego:     %10.2f Wh (%5.1f%%)              ║\n", 
               h_totals[2], (h_totals[2]/h_totals[5])*100);
        printf("║  ├─ Puerta:    %10.2f Wh (%5.1f%%)              ║\n", 
               h_totals[3], (h_totals[3]/h_totals[5])*100);
        printf("║  └─ Ascensor:  %10.2f Wh (%5.1f%%)              ║\n", 
               h_totals[4], (h_totals[4]/h_totals[5])*100);
        printf("╠═══════════════════════════════════════════════════════╣\n");
        printf("║  EVENTOS:                                             ║\n");
        printf("║  • Activaciones de puerta: %-6d                   ║\n", h_counts[1]);
        
        // Alertas
        if (h_totals[1] > h_totals[5] * 0.5) {
            printf("║  ⚠️  ALERTA: A/C consume >50%% del total              ║\n");
        }
        if (h_maxs[5] > 100.0f) {
            printf("║  ⚠️  ALERTA: Pico de consumo detectado (%.1f Wh)     ║\n", h_maxs[5]);
        }
        
        printf("╚═══════════════════════════════════════════════════════╝\n");
    }
    
    void run() {
        printf("🔄 Iniciando polling cada %d segundos...\n", POLLING_INTERVAL_SEC);
        printf("   Presiona Ctrl+C para detener\n\n");
        
        int iteration = 0;
        while (true) {
            printf("[Iteración %d]\n", ++iteration);
            
            if (fetchData()) {
                processData();
            } else {
                printf("⚠️  Error al obtener datos, reintentando...\n");
            }
            
            std::this_thread::sleep_for(std::chrono::seconds(POLLING_INTERVAL_SEC));
        }
    }
};

// ======================= Main =======================
int main() {
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║  SISTEMA DE POLLING CUDA - GOOGLE SHEETS             ║\n");
    printf("║  Smart Home Real-Time Monitoring                      ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n\n");
    
    PollingSystem system;
    system.run();
    
    return 0;
}
