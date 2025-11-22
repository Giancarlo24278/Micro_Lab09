#include "csv_handler.h"

std::vector<SensorData> loadCSV(const std::string& filename) {
    std::vector<SensorData> data;
    std::ifstream file(filename);

    if (!file.is_open()) {
        printf("  No se pudo abrir el archivo CSV: %s\n", filename.c_str());
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
    printf("CSV cargado: %zu registros\n\n", data.size());
    return data;
}