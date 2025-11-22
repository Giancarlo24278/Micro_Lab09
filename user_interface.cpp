#include "user_interface.h"
#include "common.h"
#include <iostream>
#include <algorithm>
#include <cctype>

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
    printf("\n Pregunta sobre tus datos: ");
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
    printf(" PUERTA:\n");
    printf("   • cuantas veces se abrio la puerta\n");
    printf("   • consumo de la puerta\n");
    printf("   • analiza la puerta\n\n");
    printf(" CONSUMO:\n");
    printf("   • cual es el consumo total\n");
    printf("   • cuanto gaste en total\n");
    printf("   • consumo de energia\n\n");
    printf(" LUCES:\n");
    printf("   • consumo de las luces\n");
    printf("   • cuanto gastan las luces\n\n");
    printf("  AIRE ACONDICIONADO:\n");
    printf("   • consumo del aire acondicionado\n");
    printf("   • cuanto gasta el ac\n\n");
    printf(" ESTADO:\n");
    printf("   • estado general\n");
    printf("   • resumen de la casa\n\n");
    printf(" ESTADÍSTICAS:\n");
    printf("   • dame las estadisticas\n");
    printf("   • analisis completo\n\n");
    printf("Escribe 'ayuda' para ver esto nuevamente\n");
    printf("Escribe 'salir' para terminar\n");
    printf("═══════════════════════════════════════════════════════\n");
}

void showWelcome() {
    printf("═══════════════════════════════════════════════════════\n");
    printf("    SMART HOME CSV ANALYZER CON CUDA\n");
    printf("═══════════════════════════════════════════════════════\n\n");
}