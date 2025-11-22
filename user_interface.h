#ifndef USER_INTERFACE_H
#define USER_INTERFACE_H

#include <string>

// ======================= Funciones de Interfaz =======================
bool sanitizeInput(std::string& input);
bool getUserInput(std::string& query);
void showHelp();
void showWelcome();

#endif // USER_INTERFACE_H