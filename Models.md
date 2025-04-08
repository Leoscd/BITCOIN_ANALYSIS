# Modelos de Predicción de Bitcoin

Este documento describe en detalle los modelos utilizados en el sistema de predicción de Bitcoin, su implementación, evaluación y resultados.

## Índice
- [Resumen de Modelos](#resumen-de-modelos)
- [Modelos de Clasificación](#modelos-de-clasificación)
- [Modelos de Regresión](#modelos-de-regresión)
- [Evaluación de Modelos](#evaluación-de-modelos)
- [Modelo Final](#modelo-final)
- [Interpretabilidad](#interpretabilidad)
- [Integración y Despliegue](#integración-y-despliegue)

## Resumen de Modelos

Nuestro sistema utiliza dos tipos de modelos complementarios:

1. **Modelos de Clasificación**: Predicen si ocurrirá un movimiento significativo (±3%) en las próximas 24 horas.
2. **Modelos de Regresión**: Estiman la magnitud potencial de dicho movimiento.

<div align="center">
  <img src="docs/images/model-performance-comparison.png" alt="Comparación de Rendimiento de Modelos" width="700px">
</div>

## Modelos de Clasificación

### Random Forest Classifier

**Descripción**: Ensamble de árboles de decisión entrenados en subconjuntos aleatorios de datos y características.

**Hiperparámetros Optimizados**:
```python
params = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 10,
    'min_samples_leaf': 4,
    'max_features': 'sqrt',
    'bootstrap': True,
    'class_weight': 'balanced'
}
```

**Rendimiento**:
- AUC-ROC: 0.76
- Precisión: 0.69
- Recall: 0.71
- F1-Score: 0.70

**Ventajas**:
- Buena capacidad para capturar patrones no lineales
- Robustez frente a outliers
- Baja tendencia al sobreajuste

**Desventajas**:
- Menor capacidad adaptativa en comparación con boosting
- Rendimiento subóptimo en zonas de alta volatilidad

### Gradient Boosting Classifier (XGBoost)

**Descripción**: Algoritmo de boosting que construye árboles secuencialmente, cada uno corrigiendo los errores del anterior.

**Hiperparámetros Optimizados**:
```python
params = {
    'learning_rate': 0.05,
    'n_estimators': 300,
    'max_depth': 5,
    'min_child_weight': 2,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': 1.2,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

**Rendimiento**:
- AUC-ROC: 0.80
- Precisión: 0.72
- Recall: 0.75
- F1-Score: 0.73

**Ventajas**:
- Mejor rendimiento general
- Excelente capacidad para capturar patrones complejos
- Mejor generalización

**Desventajas**:
- Mayor riesgo de sobreajuste
- Mayor tiempo de entrenamiento
- Más sensible a la configuración de hiperparámetros

### Support Vector Machine

**Descripción**: Algoritmo que busca el hiperplano que mejor separa las clases en un espacio de alta dimensionalidad.

**Hiperparámetros Optimizados**:
```python
params = {
    'C': 1.0,
    'kernel': 'rbf',
    'gamma': 'scale',
    'class_weight': 'balanced',
    'probability': True
}
```

**Rendimiento**:
- AUC-ROC: 0.74
- Precisión: 0.68
- Recall: 0.69
- F1-Score: 0.68

**Ventajas**:
- Buen rendimiento en espacios de alta dimensionalidad
- Efectivo cuando hay clara separación entre clases

**Desventajas**:
- Escalamiento inferior en comparación con los modelos basados en árboles
- Menor rendimiento general

### Modelo Ensemble

**Descripción**: Combinación ponderada de las predicciones de los modelos anteriores.

**Método de Combinación**: Promedio ponderado de probabilidades (60% XGBoost, 30% Random Forest, 10% SVM)

**Rendimiento**:
- AUC-ROC: 0.81
- Precisión: 0.73
- Recall: 0.74
- F1-Score: 0.73

**Ventajas**:
- Mejora leve respecto al mejor modelo individual
- Mayor estabilidad en las predicciones

**Desventajas**:
- Complejidad adicional
- Tiempo de inferencia incrementado

## Modelos de Regresión

### Gradient Boosting Regressor (XGBoost)

**Descripción**: Implementación de Gradient Boosting optimizada para problemas de regresión.

**Hiperparámetros Optimizados**:
```python
params = {
    'learning_rate': 0.03,
    'n_estimators': 400,
    'max_depth': 6,
    'min_child_weight': 2,
    'gamma': 0.0,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0
}
```

**Rendimiento**:
- RMSE: 2.35%
- MAE: 1.87%
- R²: 0.61

**Ventajas**:
- Mejor rendimiento general para la estimación de magnitud
- Buena capacidad para capturar relaciones no lineales

**Desventajas**:
- Tendencia a subestimar movimientos extremos
- Mayor complejidad computacional

### Elastic Net

**Descripción**: Combinación de regularización L1 (Lasso) y L2 (Ridge) para regresión lineal.

**Hiperparámetros Optimizados**:
```python
params = {
    'alpha': 0.01,
    'l1_ratio': 0.7,
    'fit_intercept': True,
    'normalize': False
}
```

**Rendimiento**:
- RMSE: 2.89%
- MAE: 2.34%
- R²: 0.48

**Ventajas**:
- Alta interpretabilidad
- Selección automática de características

**Desventajas**:
- Capacidad limitada para capturar relaciones no lineales
- Rendimiento inferior al XGBoost

### LSTM (Long Short-Term Memory)

**Descripción**: Red neuronal recurrente especializada en secuencias temporales.

**Arquitectura**:
```python
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(sequence_length, n_features)),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])
```

**Rendimiento**:
- RMSE: 2.42%
- MAE: 1.92%
- R²: 0.58

**Ventajas**:
- Buena capacidad para capturar dependencias temporales
- Detección de patrones secuenciales complejos

**Desventajas**:
- Mayor tiempo de entrenamiento
- Requiere más datos para rendimiento óptimo
- Menor estabilidad en entornos con alta volatilidad

## Evaluación de Modelos

### Métricas de Evaluación

Para una evaluación balanceada, utilizamos múltiples métricas:

#### Clasificación:
- **AUC-ROC**: Área bajo la curva ROC
- **Precisión**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: 2 * (Precisión * Recall) / (Precisión + Recall)
- **Log Loss**: Pérdida logarítmica para evaluar calibración de probabilidades

#### Regresión:
- **RMSE**: Raíz del error cuadrático medio
- **MAE**: Error absoluto medio
- **R²**: Coeficiente de determinación
- **MAPE**: Error porcentual absoluto medio (para interpretabilidad)

### Estrategia de Validación

Para asegurar la robustez de los modelos y evitar el sobreajuste, implementamos:

1. **Train-Validation-Test Split Temporal**:
   - Train: 2011-2016
   - Validation: Primera mitad de 2017
   - Test: Segunda mitad de 2017

2. **Time Series Cross-Validation**:
   - 5 folds con ventanas temporales incrementales
   - Respetando el orden cronológico de los datos

3. **Walk-Forward Validation**:
   - Re-entrenamiento periódico del modelo
   - Evaluación progresiva a medida que avanzan los datos

<div align="center">
  <img src="docs/images/time-series-cv.png" alt="Validación Cruzada Temporal" width="600px">
</div>

### Curvas de Aprendizaje

Análisis de curvas de aprendizaje para:
- Determinar si el modelo se beneficiaría de más datos
- Identificar problemas de varianza o sesgo
- Ajustar hiperparámetros de regularización

<div align="center">
  <img src="docs/images/learning-curves.png" alt="Curvas de Aprendizaje" width="600px">
</div>

## Modelo Final

Después de evaluar todos los modelos, seleccionamos **XGBoost** tanto para clasificación como para regresión como nuestro modelo final, por las siguientes razones:

1. **Rendimiento superior**: Mejor AUC-ROC (0.80) y menor error de regresión (RMSE 2.35%)
2. **Equilibrio precisión-recall**: Crucial para aplicaciones de trading
3. **Robustez**: Menor varianza entre diferentes periodos de mercado
4. **Eficiencia computacional**: Balance entre rendimiento y tiempo de inferencia

### Ajustes Finales

Para el modelo de producción, realizamos:
- Calibración de probabilidades usando Platt Scaling
- Optimización final de hiperparámetros en el conjunto completo de datos
- Entrenamiento con early stopping para determinar el número óptimo de estimadores

### Pipeline Completo

```python
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('feature_selection', SelectFromModel(lightgbm.LGBMClassifier())),
    ('classifier', XGBClassifier(**optimized_params))
])
```

## Interpretabilidad

### Importancia Global de Características

Las características más influyentes en nuestro modelo final son:

1. **volatility_ratio_5_20** (100): Ratio entre volatilidad reciente y de mediano plazo
2. **bollinger_width** (92): Ancho de las bandas de Bollinger
3. **rsi_14** (87): Relative Strength Index de 14 días
4. **volume_breakout** (85): Indicador de breakout de volumen
5. **daily_range** (80): Rango diario (High-Low)/Open
6. **macd_histogram** (75): Histograma MACD
7. **price_sma_20_ratio** (71): Ratio entre precio actual y SMA de 20 días
8. **atr_14** (70): Average True Range de 14 días
9. **volume_ratio_5_10** (68): Ratio entre volumen reciente y de mediano plazo
10. **cum_return_5d** (65): Retorno acumulado en 5 días

<div align="center">
  <img src="docs/images/feature-importance-detail.png" alt="Importancia de Características" width="700px">
</div>

### SHAP Values

Utilizamos SHAP (SHapley Additive exPlanations) para entender cómo cada característica contribuye a predicciones individuales:

<div align="center">
  <img src="docs/images/shap-summary.png" alt="SHAP Values" width="700px">
</div>

### Dependence Plots

Análisis de cómo las características clave afectan las predicciones:

<div align="center">
  <img src="docs/images/dependence-plots.png" alt="Gráficos de Dependencia" width="600px">
</div>

## Integración y Despliegue

### Serialización de Modelos

Los modelos finales se serializan usando joblib:

```python
import joblib

# Guardar modelo
joblib.dump(model, 'models/xgboost_classifier_v1.pkl')

# Cargar modelo
loaded_model = joblib.load('models/xgboost_classifier_v1.pkl')
```

### Pipeline de Predicción

El sistema completo incluye:
1. Preprocesamiento de datos en tiempo real
2. Generación de características
3. Predicción de la probabilidad de movimiento significativo
4. Predicción de la magnitud potencial
5. Generación de alertas basadas en umbrales configurables

### Actualización de Modelos

Implementamos un proceso de reciclaje periódico:
- Re-entrenamiento mensual con nuevos datos
- Evaluación continua del rendimiento
- Monitoreo de concept drift
- Ajuste automático de hiperparámetros cuando sea necesario

### Limitaciones y Consideraciones

- El modelo no captura eventos exógenos (noticias, cambios regulatorios)
- Mayor efectividad en condiciones de mercado "normales"
- El rendimiento histórico no garantiza resultados futuros
- Se recomienda usar como complemento a otros análisis, no como único criterio de decisión
