#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// ======================= Parámetros Globales =======================
constexpr int D = 8192;        // Dimensión para NLU
constexpr int K = 8;           // Número de intenciones
constexpr int MAX_QUERY = 512; // Tamaño máximo de query
constexpr int C = 7;           // Columnas CSV

// ======================= Utilidades CUDA =======================
#define CUDA_OK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

inline int ceilDiv(int a, int b) { 
    return (a + b - 1) / b; 
}

// ======================= Enumeraciones =======================
enum Intent {
    CONSUMO = 0,
    PUERTA = 1,
    LUCES = 2,
    AC = 3,
    ESTADO = 4,
    AYUDA = 5,
    ESTADISTICAS = 6,
    COMPARAR = 7
};

// ======================= Nombres de Intenciones =======================
static const char* intentNames[K] = {
    "CONSUMO", "PUERTA", "LUCES", "A/C",
    "ESTADO", "AYUDA", "ESTADISTICAS", "COMPARAR"
};

#endif // COMMON_H