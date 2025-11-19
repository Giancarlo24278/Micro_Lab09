import os
from google.colab import files

filename = "ConsumoCasaInteligente.csv"

# Eliminar el archivo anterior si existe
if os.path.isfile(filename):
    os.remove(filename)
    print(f"Archivo anterior '{filename}' eliminado.")

uploaded = files.upload()
