# Micro_Lab09

## Anotaciones:
El programa esta adaptado para correr en google collab.

### Uso
- Primero correr el código "CargarCsv.py" y subir el csv.
- Luego correr makefile y main.
- Por último: El comando para ejecutar, es el siguiente:
```
compilar: !nvcc -O3 -std=c++17 -arch=sm_75 main.cu -o chatbox_cuda
correr: !./chatbox_cuda
```
- Al correr el programa, en consola escribir "ayuda" para obtener instrucciones de uso.
