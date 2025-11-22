// CC3086 - Lab 9: Smart Home Chat-Box con IA (CUDA + CSV) - Versión Modular

#include "common.h"
#include "csv_handler.h"
#include "nlu_engine.h"
#include "data_analyzer.h"
#include "user_interface.h"

int main() {
    showWelcome();

    // Cargar datos CSV
    const std::string csv_path = "ConsumoCasaInteligente.csv";
    std::vector<SensorData> csvData = loadCSV(csv_path);

    if (csvData.empty()) {
        printf(" No se encontró el archivo CSV o está vacío.\n");
        printf("   Coloca el archivo en el mismo directorio que el ejecutable.\n");
        printf("   Nombre esperado: ConsumoCasaInteligente.csv\n\n");
        return 1;
    }

    // Inicializar motor NLU
    NLUEngine nluEngine;
    nluEngine.initialize();

    printf("Sistema listo. Escribe 'ayuda' para ver comandos.\n");

    int queryCount = 0;
    std::string userQuery;

    // Bucle principal
    while (true) {
        if (!getUserInput(userQuery)) {
            printf("\n Cerrando sistema...\n");
            break;
        }

        queryCount++;

        // Mostrar ayuda si se solicita
        if (userQuery == "ayuda" || userQuery == "help") {
            showHelp();
            continue;
        }

        // Procesar query con NLU
        int topIntent = 0;
        float confidence = 0.0f;
        float processingTime = 0.0f;

        nluEngine.processQuery(userQuery, topIntent, confidence, processingTime);

        printf("\n Intent: %s (%.1f%% confianza)\n",
               intentNames[topIntent], confidence * 100);

        // Analizar datos basado en el intent
        analyzeData(csvData, topIntent, userQuery);

        printf("\n  Tiempo de procesamiento: %.2f ms\n", processingTime);
    }

    // Limpieza
    nluEngine.cleanup();

    printf("\n✓ Sesión finalizada: %d consultas procesadas\n", queryCount);
    return 0;
}