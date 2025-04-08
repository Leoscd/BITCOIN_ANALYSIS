# Metodología del Análisis de Bitcoin

Este documento describe en detalle el enfoque metodológico empleado en nuestro sistema de análisis y predicción de movimientos del Bitcoin.

## Índice
- [Definición del Problema](#definición-del-problema)
- [Enfoque General](#enfoque-general)
- [Datos y Preprocesamiento](#datos-y-preprocesamiento)
- [Ingeniería de Características](#ingeniería-de-características)
- [Modelado](#modelado)
- [Validación](#validación)
- [Interpretación y Aplicación](#interpretación-y-aplicación)

## Definición del Problema

El proyecto aborda dos problemas complementarios:

1. **Problema de Clasificación**: Predecir si el Bitcoin experimentará un movimiento significativo (definido como una variación de ±3% o más) en las próximas 24 horas.

2. **Problema de Regresión**: Estimar la magnitud potencial de dicho movimiento, expresada como porcentaje de cambio respecto al precio actual.

La combinación de estos enfoques permite no solo anticipar cuándo podrían ocurrir movimientos relevantes, sino también su magnitud aproximada, información crucial para estrategias de trading o gestión de riesgos.

## Enfoque General

Nuestra metodología sigue un enfoque híbrido que combina:

- **Análisis Técnico Tradicional**: Utilizando indicadores establecidos en el análisis de mercados financieros
- **Machine Learning Avanzado**: Empleando algoritmos capaces de capturar patrones complejos y no lineales
- **Series Temporales**: Respetando la naturaleza secuencial de los datos financieros
- **Validación Temporal**: Asegurando que el modelo se evalúe en un contexto realista de predicción

<div align="center">
  <img src="docs/images/methodology-diagram.png" alt="Metodología" width="700px">
</div>

## Datos y Preprocesamiento

### Fuentes de Datos
Utilizamos datos históricos de Bitcoin que abarcan el período 2011-2017, incluyendo:
- Precios OHLC (Open, High, Low, Close)
- Volumen de transacciones
- Datos de blockchain (número de transacciones, hashrate, etc.)

### Preprocesamiento
1. **Limpieza de Datos**:
   - Manejo de valores faltantes mediante interpolación
   - Detección y tratamiento de outliers usando IQR
   - Normalización de fechas y ajuste de zonas horarias

2. **Transformaciones Iniciales**:
   - Aplicación de logaritmos a precios para estabilizar varianza
   - Normalización de volúmenes para comparabilidad entre periodos
   - Cálculo de retornos diarios, semanales y mensuales

3. **Creación de Variable Objetivo**:
   - Para clasificación: Variable binaria que indica si el movimiento en las próximas 24h excede el umbral de ±3%
   - Para regresión: Porcentaje de cambio en las próximas 24h

## Ingeniería de Características

Implementamos más de 50 características derivadas de los datos históricos, agrupadas en:

### Indicadores de Tendencia
- Medias móviles (simples y exponenciales) de 5, 10, 20, 50 y 200 días
- MACD (Moving Average Convergence Divergence)
- Índice direccional (ADX)
- Ratios de precios entre diferentes períodos

### Indicadores de Volatilidad
- Bandas de Bollinger (20 días, 2 desviaciones estándar)
- ATR (Average True Range) de 14 días
- Volatilidad histórica: desviación estándar de retornos (5, 10, 20 días)
- Volatilidad intradiaria (High-Low/Open)

### Indicadores de Momentum
- RSI (Relative Strength Index) de 14 días
- Estocástico (%K, %D)
- ROC (Rate of Change) de 10 días
- OBV (On-Balance Volume)

### Indicadores de Soporte/Resistencia
- Distancia a máximos/mínimos recientes (5, 10, 20, 50 días)
- Niveles de Fibonacci calculados sobre rangos recientes
- Cálculo de soportes y resistencias dinámicos

### Variables Temporales
- Día de la semana
- Indicadores de fin de mes/trimestre
- Estacionalidad histórica

### Variables de Blockchain
- Crecimiento de transacciones
- Dificultad de minado
- Hashrate

## Modelado

### Selección de Características
1. **Análisis de Correlación**:
   - Eliminación de variables altamente correlacionadas (>0.85)
   - Análisis de multicolinealidad mediante VIF

2. **Selección Basada en Importancia**:
   - Uso de Random Forest para estimación inicial de importancia
   - Selección de las 20-25 características más relevantes

### Modelos Evaluados

#### Para Clasificación
- Random Forest
- Gradient Boosting (XGBoost)
- Support Vector Machines
- Redes Neuronales
- Ensemble de múltiples modelos

#### Para Regresión
- Gradient Boosting (XGBoost)
- Elastic Net
- SVR (Support Vector Regression)
- LSTM (Long Short-Term Memory)

### Optimización de Hiperparámetros
- Grid Search con validación cruzada temporal
- Bayesian Optimization para modelos complejos
- Priorización de recall para el problema de clasificación

## Validación

### Estrategia de Validación Temporal
Para respetar la naturaleza secuencial de los datos financieros:
- **Train**: Datos de 2011-2016
- **Validation**: Primera mitad de 2017
- **Test**: Segunda mitad de 2017

### Cross-Validation Especial
- **Time-series split**: 5 folds con ventanas temporales incrementales
- **Purged cross-validation**: Eliminación de muestras cercanas entre train/test para evitar fugas de información

### Métricas de Evaluación

#### Para Clasificación
- AUC-ROC: 0.80 (modelo final)
- Precisión: 0.72
- Recall: 0.75
- F1-Score: 0.73

#### Para Regresión
- RMSE: 2.35%
- MAE: 1.87%
- R²: 0.61

### Backtesting
Simulación de estrategias de trading basadas en las predicciones del modelo:
- Evaluación de rentabilidad vs. estrategia buy-and-hold
- Análisis de drawdown y sharpe ratio
- Sensibilidad a costos de transacción

## Interpretación y Aplicación

### Interpretabilidad del Modelo
- Uso de SHAP values para explicar predicciones individuales
- Partial dependence plots para entender relaciones no lineales
- Global feature importance para identificar drivers principales

### Sistema de Alertas
Las predicciones se traducen en un sistema de alertas con:
- Niveles de confianza (alta, media, baja)
- Dirección estimada (alcista/bajista)
- Horizonte temporal (24h, 72h)
- Magnitud esperada

### Limitaciones
- El modelo no captura eventos exógenos (noticias, cambios regulatorios)
- Mayor efectividad en condiciones de mercado "normales"
- Necesidad de recalibración periódica (cada 3-6 meses)

### Aplicaciones Recomendadas
- Complemento a análisis técnico tradicional
- Identificación de oportunidades potenciales de trading
- Gestión de riesgo en posiciones de Bitcoin
- Timing para entradas/salidas en estrategias de inversión
