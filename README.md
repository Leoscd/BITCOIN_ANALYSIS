# Análisis y Predicción de Bitcoin con Machine Learning

[![MIT License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/Leoscd/BITCOIN_ANALYSIS/graphs/commit-activity)
## Descripción

El proyecto utiliza datos históricos de Bitcoin (2011-2017) para construir modelos predictivos capaces de:

1. **Clasificar** movimientos significativos en el precio (AUC 0.80)
2. **Estimar** la magnitud potencial de dichos movimientos

Mediante la combinación de análisis técnico tradicional y algoritmos modernos de machine learning, el sistema identifica patrones que preceden a movimientos relevantes en el mercado de criptomonedas.

## Características

- **Feature Engineering** especializado para series temporales financieras
- **Indicadores técnicos** personalizados (volatilidad, Bollinger Bands, etc.)
- **Detección de breakouts** con confirmación por volumen
- **Modelos predictivos** de clasificación y regresión
- **Validación temporal** respetando la naturaleza secuencial de los datos
- **Pipeline de procesamiento** modular y extensible

## Estructura del Proyecto

```
bitcoin-analysis/
├── data/                    # Datos históricos y procesados
├── notebooks/               # Jupyter notebooks del análisis
├── models/                  # Modelos entrenados guardados
├── src/                     # Código fuente 
│   ├── features/            # Generación de características
│   ├── models/              # Implementación de modelos
│   └── utils/               # Funciones auxiliares
├── README.md                # Este archivo
├── Methodology.md           # Explicación metodológica detallada
├── Feature_Description.md   # Catálogo de variables utilizadas
├── Model_Evaluation.md      # Resultados y evaluación de modelos
├── requirements.txt         # Dependencias del proyecto
├── CONTRIBUTING.md          # Guía para contribuciones
└── LICENSE                  # Licencia MIT
```

## Resultados Principales

- **Modelo de clasificación**: AUC de 0.80 en la identificación de movimientos significativos
- **Variables más predictivas**: Medidas de volatilidad (intraday, 5d, 20d) y ratios de precios
- **Modelo optimizado**: Gradient Boosting con hiperparámetros específicamente ajustados
- **Capacidad predictiva**: Identificación efectiva de oportunidades potenciales de trading

## Instalación y Uso

1. Clonar el repositorio:
```bash
git clone https://github.com/Leoscd/BITCOIN_ANALYSIS.git
cd BITCOIN_ANALYSIS
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecutar notebooks para reproducir el análisis o utilizar los módulos para predicciones:
```python
from src.models import BitcoinPredictor
from src.features import TechnicalFeatures

# Cargar datos
import pandas as pd
data = pd.read_csv('data/bitcoin_historical.csv')

# Generar características
features = TechnicalFeatures(data).generate_all()

# Cargar modelo pre-entrenado
predictor = BitcoinPredictor.load('models/gradient_boosting.pkl')

# Realizar predicciones
predictions = predictor.predict(features)
```

## Documentación Adicional

- [Metodología](Methodology.md): Enfoque técnico detallado del proyecto
- [Descripción de Variables](Feature_Description.md): Catálogo completo de características implementadas
- [Evaluación de Modelos](Model_Evaluation.md): Resultados, métricas y análisis de rendimiento

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

## Contribuciones

Las contribuciones son bienvenidas. Consulta [CONTRIBUTING.md](CONTRIBUTING.md) para conocer las pautas de contribución.

---

Desarrollado por [Leonardo] | [www.linkedin.com/in/leo-iml]


