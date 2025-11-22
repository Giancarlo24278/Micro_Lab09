Smart Home CSV Analyzer

Sistema de análisis de consumo energético con procesamiento NLU en CUDA.
Requisitos

    CUDA Toolkit (versión 11.0 o superior)
    GPU con compute capability 7.5 o superior
    GNU Make
    C++14 compatible compiler

Compilación
Usando Makefile

# Compilar el proyecto
make
# Limpiar archivos objeto y ejecutable
make clean
# Limpiar y recompilar
make rebuild
# Compilar y ejecutar
make run

Asegúrate de que el archivo ConsumoCasaInteligente.csv esté en el mismo directorio.
Módulos del Sistema
1. common.h

    Constantes globales (D, K, MAX_QUERY)
    Macros CUDA (CUDA_OK)
    Enumeración de intenciones
    Utilidades comunes

2. csv_handler (h/cpp)

    Estructura SensorData
    Función loadCSV() para cargar datos

3. nlu_kernels (cuh/cu)

    tokenize3grams: Tokenización con hash 3-gramas
    l2normalize: Normalización L2
    matvecDotCos: Producto matriz-vector
    fuseDecision: Selección de intención (argmax)

4. nlu_engine (h/cu)

    Clase NLUEngine
    Gestión de recursos CUDA
    Procesamiento de queries
    Inicialización de prototipos

5. data_analyzer (h/cpp)

    Función analyzeData()
    Análisis por tipo de intención
    Cálculo de estadísticas

6. user_interface (h/cpp)

    getUserInput(): Captura entrada usuario
    sanitizeInput(): Validación entrada
    showHelp(): Ayuda sistema
    showWelcome(): Mensaje bienvenida

7. main.cu

    Orquestación del sistema
    Bucle principal de interacción
    Gestión del flujo del programa

Comandos Disponibles

    ayuda / help - Muestra ayuda
    salir / exit - Termina el programa
    Preguntas en lenguaje natural sobre consumo energético

Ajuste de Parámetros

Edita common.h para modificar:

    D: Dimensión vectores (default: 8192)
    K: Número de intenciones (default: 8)
    MAX_QUERY: Longitud máxima query (default: 512)

Notas

    Ajusta -arch=sm_XX en el Makefile según tu GPU
    El sistema usa memoria pinned para transfers rápidos
    Implementa streams CUDA para overlapping
    En el siguiente link puedes ver un ejemplo implementación en google collab: https://colab.research.google.com/drive/1coOUo3VKRwqS7JfElEv0dOeUtbZ394T1?usp=sharing
