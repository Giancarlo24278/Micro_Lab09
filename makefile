# Makefile para Smart Home CSV Analyzer

# Compilador
NVCC = nvcc

# Flags de compilación
NVCC_FLAGS = -std=c++14 -O3 -arch=sm_75 --extended-lambda
CXX_FLAGS = -std=c++14 -O3

# Archivos objeto
OBJS = main.o nlu_engine.o nlu_kernels.o csv_handler.o data_analyzer.o user_interface.o

# Ejecutable
TARGET = smart_home_analyzer

# Regla principal
all: $(TARGET)

# Linkear todos los objetos
$(TARGET): $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Compilar main
main.o: main.cu common.h csv_handler.h nlu_engine.h data_analyzer.h user_interface.h
	$(NVCC) $(NVCC_FLAGS) -c main.cu

# Compilar motor NLU
nlu_engine.o: nlu_engine.cu nlu_engine.h nlu_kernels.cuh common.h
	$(NVCC) $(NVCC_FLAGS) -c nlu_engine.cu

# Compilar kernels
nlu_kernels.o: nlu_kernels.cu nlu_kernels.cuh common.h
	$(NVCC) $(NVCC_FLAGS) -c nlu_kernels.cu

# Compilar módulos C++
csv_handler.o: csv_handler.cpp csv_handler.h
	$(NVCC) $(CXX_FLAGS) -c csv_handler.cpp

data_analyzer.o: data_analyzer.cpp data_analyzer.h csv_handler.h common.h
	$(NVCC) $(CXX_FLAGS) -c data_analyzer.cpp

user_interface.o: user_interface.cpp user_interface.h common.h
	$(NVCC) $(CXX_FLAGS) -c user_interface.cpp

# Limpiar archivos generados
clean:
	rm -f $(OBJS) $(TARGET)

# Limpiar y recompilar
rebuild: clean all

# Ejecutar
run: $(TARGET)
	./$(TARGET)

# Profile con nvprof (legacy) o ncu (moderno)
profile: $(TARGET)
	@echo "Perfilando con nvprof..."
	nvprof --print-gpu-trace ./$(TARGET) 2>&1 || \
	ncu --target-processes all ./$(TARGET)

.PHONY: all clean rebuild run profile