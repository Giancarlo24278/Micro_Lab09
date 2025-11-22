#include "data_analyzer.h"
#include <algorithm>
#include <cstdio>

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

    float max_val = std::max({total_luces, total_ac, total_riego, 
                              total_puerta, total_ascensor});

    switch(intent) {
        case PUERTA:
            printf(" Análisis de Puerta:\n");
            printf("   • Activaciones detectadas: %.0f eventos\n", count_puerta);
            printf("   • Consumo total: %.2f Wh\n", total_puerta);
            printf("   • Consumo máximo por evento: %.2f Wh\n", max_puerta);
            printf("   • Promedio por activación: %.2f Wh\n",
                   count_puerta > 0 ? total_puerta/count_puerta : 0);
            break;

        case CONSUMO:
            printf(" Consumo Total del Sistema:\n");
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
            printf(" Análisis de Iluminación:\n");
            printf("   • Consumo total: %.2f Wh\n", total_luces);
            printf("   • Porcentaje del total: %.1f%%\n",
                   (total_luces/total_consumo)*100);
            break;

        case AC:
            printf("  Análisis de Aire Acondicionado:\n");
            printf("   • Consumo total: %.2f Wh\n", total_ac);
            printf("   • Porcentaje del total: %.1f%%\n",
                   (total_ac/total_consumo)*100);
            if (total_ac > total_consumo * 0.5) {
                printf("El A/C representa más del 50%% del consumo\n");
            }
            break;

        case ESTADO:
            printf(" Estado General del Sistema:\n");
            printf("   • Total de registros: %zu\n", data.size());
            printf("   • Consumo acumulado: %.2f Wh\n", total_consumo);
            printf("   • Sistema más utilizado: ");

            if (max_val == total_luces) printf("Luces");
            else if (max_val == total_ac) printf("A/C");
            else if (max_val == total_riego) printf("Riego");
            else if (max_val == total_puerta) printf("Puerta");
            else printf("Ascensor");
            printf(" (%.2f Wh)\n", max_val);
            break;

        case ESTADISTICAS:
            printf(" Estadísticas Detalladas:\n");
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