#ifndef CSV_HANDLER_H
#define CSV_HANDLER_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <cstdio>

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

// ======================= Funciones CSV =======================
std::vector<SensorData> loadCSV(const std::string& filename);

#endif // CSV_HANDLER_H