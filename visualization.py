# visualization.py
# Visualización de Datos con Matplotlib/Plotly

import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

class DataVisualizer:
    def __init__(self, csv_url):
        self.csv_url = csv_url
        self.df = None
        
    def load_data(self):
        """Cargar datos desde Google Sheets"""
        try:
            self.df = pd.read_csv(self.csv_url)
            print(f"✓ Datos cargados: {len(self.df)} registros")
            return True
        except Exception as e:
            print(f"❌ Error al cargar datos: {e}")
            return False
    
    def plot_consumption_overview(self):
        """Gráfico de consumo general"""
        if self.df is None:
            print("⚠️  Carga los datos primero")
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Consumo Total en el Tiempo', 
                          'Distribución por Dispositivo',
                          'Consumo Promedio por Dispositivo',
                          'Actividad de Puerta'),
            specs=[[{"type": "scatter"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Consumo total en el tiempo
        fig.add_trace(
            go.Scatter(x=self.df.index, y=self.df['Total (Wh)'],
                      mode='lines', name='Total',
                      line=dict(color='#FF6B6B', width=2)),
            row=1, col=1
        )
        
        # 2. Pie chart de distribución
        totals = {
            'Luces': self.df['Luces (Wh)'].sum(),
            'A/C': self.df['A/C (Wh)'].sum(),
            'Riego': self.df['Riego (Wh)'].sum(),
            'Puerta': self.df['Puerta (Wh)'].sum(),
            'Ascensor': self.df['Ascensor (Wh)'].sum()
        }
        
        fig.add_trace(
            go.Pie(labels=list(totals.keys()), values=list(totals.values()),
                  marker=dict(colors=['#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DFE6E9'])),
            row=1, col=2
        )
        
        # 3. Barras de promedio
        averages = {k: v/len(self.df) for k, v in totals.items()}
        fig.add_trace(
            go.Bar(x=list(averages.keys()), y=list(averages.values()),
                  marker=dict(color='#74B9FF')),
            row=2, col=1
        )
        
        # 4. Actividad de puerta
        puerta_events = self.df[self.df['Puerta (Wh)'] > 0.1]
        fig.add_trace(
            go.Scatter(x=puerta_events.index, y=puerta_events['Puerta (Wh)'],
                      mode='markers', name='Activaciones',
                      marker=dict(size=8, color='#FD79A8')),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="Dashboard de Consumo Energético",
            showlegend=True,
            height=800
        )
        
        fig.show()
    
    def plot_heatmap(self):
        """Mapa de calor por hora y dispositivo"""
        if self.df is None:
            return
        
        # Agregar columna de hora (simulada desde timestamp)
        self.df['Hora'] = (self.df['Timestamp (s)'] // 3600) % 24
        
        devices = ['Luces (Wh)', 'A/C (Wh)', 'Riego (Wh)', 'Puerta (Wh)', 'Ascensor (Wh)']
        hourly_avg = self.df.groupby('Hora')[devices].mean()
        
        fig = go.Figure(data=go.Heatmap(
            z=hourly_avg.T.values,
            x=hourly_avg.index,
            y=[d.replace(' (Wh)', '') for d in devices],
            colorscale='RdYlBu_r',
            text=hourly_avg.T.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title='Mapa de Calor: Consumo Promedio por Hora',
            xaxis_title='Hora del Día',
            yaxis_title='Dispositivo',
            height=500
        )
        
        fig.show()
    
    def plot_anomalies(self, threshold=50):
        """Detectar y visualizar anomalías"""
        if self.df is None:
            return
        
        anomalies = self.df[self.df['Total (Wh)'] > threshold]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.df.index,
            y=self.df['Total (Wh)'],
            mode='lines',
            name='Consumo Normal',
            line=dict(color='lightblue', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies['Total (Wh)'],
            mode='markers',
            name=f'Anomalías (>{threshold} Wh)',
            marker=dict(size=10, color='red', symbol='x')
        ))
        
        fig.add_hline(y=threshold, line_dash="dash", 
                     line_color="orange",
                     annotation_text="Umbral")
        
        fig.update_layout(
            title=f'Detección de Anomalías en Consumo (>{threshold} Wh)',
            xaxis_title='Registro',
            yaxis_title='Consumo Total (Wh)',
            height=500
        )
        
        fig.show()
        
        print(f"\n📊 Estadísticas de Anomalías:")
        print(f"   Total detectadas: {len(anomalies)}")
        print(f"   Consumo promedio anómalo: {anomalies['Total (Wh)'].mean():.2f} Wh")
        print(f"   Consumo máximo: {anomalies['Total (Wh)'].max():.2f} Wh")
    
    def generate_report(self):
        """Generar reporte completo"""
        if self.df is None:
            return
        
        print("\n" + "="*60)
        print("REPORTE DE ANÁLISIS DE DATOS")
        print("="*60)
        
        print(f"\n📈 Estadísticas Generales:")
        print(f"   • Total de registros: {len(self.df)}")
        print(f"   • Periodo: {self.df['Timestamp (s)'].min():.0f}s - {self.df['Timestamp (s)'].max():.0f}s")
        print(f"   • Consumo total acumulado: {self.df['Total (Wh)'].sum():.2f} Wh")
        
        print(f"\n⚡ Consumo por Dispositivo:")
        devices = {
            'Luces': self.df['Luces (Wh)'].sum(),
            'A/C': self.df['A/C (Wh)'].sum(),
            'Riego': self.df['Riego (Wh)'].sum(),
            'Puerta': self.df['Puerta (Wh)'].sum(),
            'Ascensor': self.df['Ascensor (Wh)'].sum()
        }
        
        total = sum(devices.values())
        for device, consumption in sorted(devices.items(), key=lambda x: x[1], reverse=True):
            percentage = (consumption / total * 100) if total > 0 else 0
            print(f"   • {device:12s}: {consumption:10.2f} Wh ({percentage:5.1f}%)")
        
        print(f"\n🚪 Eventos de Puerta:")
        puerta_events = len(self.df[self.df['Puerta (Wh)'] > 0.1])
        print(f"   • Activaciones detectadas: {puerta_events}")
        print(f"   • Consumo promedio por activación: {devices['Puerta']/puerta_events:.2f} Wh" if puerta_events > 0 else "   • No hay activaciones")
        
        print(f"\n⚠️  Alertas:")
        if devices['A/C'] > total * 0.5:
            print(f"   • ALTA: A/C consume {devices['A/C']/total*100:.1f}% del total")
        if self.df['Total (Wh)'].max() > 100:
            print(f"   • ALTA: Pico de consumo detectado: {self.df['Total (Wh)'].max():.2f} Wh")
        
        print("="*60 + "\n")

# ======================= Ejemplo de Uso =======================
if __name__ == "__main__":
    CSV_URL = "https://docs.google.com/spreadsheets/d/1d3RnoKpjoudoDzbbDu_d0ED5AwQKYIWqx271EEZSNaU/export?format=csv"
    
    viz = DataVisualizer(CSV_URL)
    
    if viz.load_data():
        print("\n🎨 Generando visualizaciones...\n")
        
        viz.generate_report()
        viz.plot_consumption_overview()
        viz.plot_heatmap()
        viz.plot_anomalies(threshold=50)
        
        print("✓ Visualización completa")
