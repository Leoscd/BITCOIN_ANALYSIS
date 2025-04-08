# Análisis y Predicción de Bitcoin con Machine Learning

<div align="center">
  <img src="docs/images/bitcoin-banner.png" alt="Bitcoin Analysis Banner" width="600px">
</div>

<div align="center">
  <img src="https://img.shields.io/badge/ML-Bitcoin%20Analysis-orange" alt="ML Badge">
  <img src="https://img.shields.io/badge/Python-3.7%2B-blue" alt="Python Badge">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange" alt="Jupyter Badge">
  <img src="https://img.shields.io/badge/Status-Active-green" alt="Status Badge">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License Badge">
</div>

## 🔍 Descripción

Este proyecto utiliza datos históricos de Bitcoin (2011-2017) para construir modelos predictivos capaces de:

- **Clasificar movimientos significativos** en el precio (AUC 0.80)
- **Estimar la magnitud potencial** de dichos movimientos
- **Generar alertas** basadas en patrones identificados

Mediante la combinación de análisis técnico tradicional y algoritmos modernos de machine learning, el sistema identifica patrones que preceden a movimientos relevantes en el mercado de criptomonedas.

## ✨ Características Principales

- **Feature Engineering especializado** para series temporales financieras
- **Indicadores técnicos personalizados** (volatilidad, Bollinger Bands, etc.)
- **Detección de breakouts** con confirmación por volumen
- **Modelos predictivos** de clasificación y regresión
- **Validación temporal** respetando la naturaleza secuencial de los datos
- **Pipeline de procesamiento** modular y extensible

<div align="center">
  <img src="docs/images/prediction-chart.png" alt="Prediction Chart" width="700px">
</div>

## 📊 Resultados Principales

- **Modelo de clasificación**: AUC de 0.80 en la identificación de movimientos significativos
- **Variables más predictivas**: Medidas de volatilidad (intraday, 5d, 20d) y ratios de precios
- **Modelo optimizado**: Gradient Boosting con hiperparámetros específicamente ajustados
- **Capacidad predictiva**: Identificación efectiva de oportunidades potenciales de trading

<div align="center">
  <img src="docs/images/feature-importance.png" alt="Feature Importance" width="600px">
</div>

## 🏗️ Estructura del Proyecto

```
bitcoin-analysis/
├── data/                    # Datos históricos y procesados
├── notebooks/               # Jupyter notebooks del análisis
├── models/                  # Modelos entrenados guardados
├── src/                     # Código fuente 
│   ├── features/            # Generación de características
│   ├── models/              # Implementación de modelos
│   └── utils/               # Funciones auxiliares
├── docs/                    # Documentación detallada
│   ├── images/              # Imágenes para documentación
│   └── examples/            # Ejemplos de uso
├── README.md                # Este archivo
├── METHODOLOGY.md           # Explicación metodológica detallada
├── FEATURES.md              # Catálogo de variables utilizadas
├── MODELS.md                # Resultados y evaluación de modelos
├── requirements.txt         # Dependencias del proyecto
├── CONTRIBUTING.md          # Guía para contribuciones
└── LICENSE                  # Licencia MIT
```

## 🚀 Instalación y Uso

### Requisitos Previos
- Python 3.7+
- pandas, numpy, scikit-learn, matplotlib, seaborn
- Jupyter Notebook (opcional, para exploración)

### Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/Leoscd/BITCOIN_ANALYSIS.git
   cd BITCOIN_ANALYSIS
   ```

2. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Ejemplo de Uso

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

## 📚 Documentación Adicional

- [Metodología](METHODOLOGY.md): Enfoque técnico detallado del proyecto
- [Descripción de Variables](FEATURES.md): Catálogo completo de características implementadas
- [Evaluación de Modelos](MODELS.md): Resultados, métricas y análisis de rendimiento

## 🛠️ Tecnologías Utilizadas

- **Lenguajes**: Python
- **Análisis de Datos**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost
- **Visualización**: matplotlib, seaborn, plotly
- **Series Temporales**: statsmodels, pmdarima

## 🧪 Validación y Pruebas

Los modelos se validaron utilizando:
- **Validación temporal**: Train en periodo 2011-2016, Test en 2017
- **Cross-validation con walk-forward**: Respetando la naturaleza temporal de los datos
- **Backtesting**: Simulación de estrategias de trading basadas en las predicciones

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, revisa primero [nuestra guía de contribución](CONTRIBUTING.md).

1. Fork el repositorio
2. Crea una nueva rama (`git checkout -b feature/nueva-caracteristica`)
3. Haz commit de tus cambios (`git commit -m 'Añadida nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Abre un Pull Request

## 📬 Contacto

Leonardo Díaz - [GitHub Profile](https://github.com/Leoscd) - [LinkedIn](https://www.linkedin.com/in/leonardoadriandiaz/)

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - consulta el archivo [LICENSE](LICENSE) para más detalles.
