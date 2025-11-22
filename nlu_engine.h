#ifndef NLU_ENGINE_H
#define NLU_ENGINE_H

#include "common.h"
#include <vector>
#include <string>

// ======================= Clase NLU Engine =======================
class NLUEngine {
private:
    cudaStream_t stream;
    cudaEvent_t evStart, evStop;
    
    float *dM;           // Matriz de prototipos
    char *hQ, *dQ;       // Query host/device
    float *dVQ;          // Vector query
    float *dScores;      // Scores
    float *hScores;      // Scores host
    int *dTop;           // Top intent
    float *dConfidence;  // Confianza
    
    std::vector<float> hM; // Prototipos host

public:
    NLUEngine();
    ~NLUEngine();
    
    void initialize();
    void processQuery(const std::string& query, int& topIntent, 
                      float& confidence, float& processingTime);
    void cleanup();

private:
    void initIntentPrototypes();
};

#endif // NLU_ENGINE_H