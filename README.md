# Sistema Predictivo de Movimientos Significativos en Bitcoin

## Descripción
Este proyecto implementa un sistema de machine learning para predecir movimientos significativos en el precio de Bitcoin, utilizando indicadores técnicos, análisis de volatilidad y patrones de volumen. El modelo alcanza un AUC de 0.80, permitiendo identificar oportunidades de trading con alta precisión.

## Características Principales
- **Feature Engineering avanzado**: Creación de indicadores de volatilidad, Bandas de Bollinger y métricas de volumen normalizado
- **Detección de Breakouts**: Identificación automática de fugas de precios con clasificación por calidad (Fuerte, Débil, Neutral)
- **Análisis de Confirmación**: Desarrollo de indicadores combinados que integran señales de precio y volumen
- **Modelos Predictivos**: Implementación de clasificación binaria para detectar movimientos significativos y estimación de magnitud

## Estructura del Proyecto
- `notebooks/`: Jupyter notebooks con el desarrollo completo
  - `feature_enginner_bitcoin.ipynb`: Creación y selección de variables
  - `bitcoin_modelados.ipynb`: Entrenamiento y evaluación de modelos
- `src/`: Scripts modulares para procesamiento de datos
  - `bitcoin_preprocessor.py`: Limpieza y preparación de datos
  - `bitcoin_visualization.py`: Visualizaciones para análisis técnico

## Resultados
El sistema desarrolla dos modelos complementarios:
- **Modelo de Clasificación**: Detecta movimientos significativos con 80% de precisión (AUC)
- **Modelo de Regresión**: Estima la magnitud de retornos esperados

Las "Fugas Fuertes" identificadas por el algoritmo presentan un retorno promedio del 16.62%, significativamente superior a los movimientos normales del mercado.

## Tecnologías Utilizadas
- Python 3.8+
- Pandas, NumPy para manipulación de datos
- Scikit-learn para modelado predictivo
- Matplotlib, Seaborn para visualizaciones
- Feature engineering personalizado para datos financieros

## Implementación
El sistema puede implementarse como:
1. Herramienta de alerta temprana para traders
2. Componente para estrategias algorítmicas
3. Base para sistemas avanzados de gestión de riesgo

## Capturas de Pantalla
[Incluye imágenes de tus visualizaciones más impactantes]

## Contacto
[Tu información de contacto o enlace a LinkedIn]
