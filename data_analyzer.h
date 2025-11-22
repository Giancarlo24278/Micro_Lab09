#ifndef DATA_ANALYZER_H
#define DATA_ANALYZER_H

#include "csv_handler.h"
#include "common.h"
#include <string>

// ======================= Funciones de Análisis =======================
void analyzeData(const std::vector<SensorData>& data, 
                 int intent, 
                 const std::string& query);

#endif // DATA_ANALYZER_H