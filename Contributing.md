# Guía de Contribución

¡Gracias por tu interés en contribuir al proyecto de Análisis y Predicción de Bitcoin! Este documento proporciona lineamientos para contribuir al proyecto de manera efectiva.

## Índice
- [Código de Conducta](#código-de-conducta)
- [¿Cómo Puedo Contribuir?](#cómo-puedo-contribuir)
- [Flujo de Trabajo con Git](#flujo-de-trabajo-con-git)
- [Estilo de Código](#estilo-de-código)
- [Pruebas](#pruebas)
- [Documentación](#documentación)

## Código de Conducta

Este proyecto y todos sus participantes están regidos por un Código de Conducta que promueve un entorno abierto, respetuoso e inclusivo. Al participar, se espera que respetes este código.

## ¿Cómo Puedo Contribuir?

### Reportando Bugs

Si encuentras un bug, por favor crea un issue con la siguiente información:
- Título descriptivo del problema
- Pasos detallados para reproducir el bug
- Comportamiento esperado vs. comportamiento observado
- Capturas de pantalla si aplica
- Entorno (sistema operativo, versión de Python, etc.)

### Sugiriendo Mejoras

Las sugerencias de mejoras son bienvenidas. Por favor crea un issue con:
- Título claro de la propuesta
- Descripción detallada de la mejora
- Justificación de por qué esta mejora sería valiosa
- Ejemplos o mockups si es posible

### Añadiendo Nuevas Características

Para añadir nuevas características al proyecto:
1. **Indicadores Técnicos**: Implementaciones de indicadores financieros adicionales
2. **Modelos Predictivos**: Algoritmos alternativos o mejoras a los existentes
3. **Visualizaciones**: Nuevas formas de visualizar datos o resultados
4. **Optimizaciones**: Mejoras de rendimiento o eficiencia

### Mejorando la Documentación

La documentación clara es esencial. Puedes contribuir con:
- Correcciones en la documentación existente
- Ampliación de explicaciones técnicas
- Adición de ejemplos de uso
- Traducción a otros idiomas

## Flujo de Trabajo con Git

1. **Fork** el repositorio
2. **Clona** tu fork: `git clone https://github.com/tu-usuario/BITCOIN_ANALYSIS.git`
3. **Añade el repositorio original como remote**: `git remote add upstream https://github.com/Leoscd/BITCOIN_ANALYSIS.git`
4. **Crea una rama** para tu contribución: `git checkout -b feature/mi-caracteristica`
5. **Realiza tus cambios** y haz commits con mensajes descriptivos
6. **Mantén tu fork actualizado**: 
   ```
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```
7. **Envía un Pull Request** con una descripción clara de los cambios

### Convenciones de Commit

Utiliza mensajes de commit que describan claramente el propósito:
- `feature: añadido nuevo indicador RSI personalizado`
- `fix: corregido cálculo de bandas de Bollinger`
- `docs: actualizada documentación de modelos`
- `test: añadidos tests para feature engineering`
- `refactor: optimizado código de preprocesamiento`

## Estilo de Código

### Python

- Sigue [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Utiliza docstrings en formato [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Nombres descriptivos para variables y funciones
- Mantén las funciones pequeñas y con un solo propósito

```python
def calculate_bollinger_bands(prices, window=20, num_std=2):
    """
    Calcula las bandas de Bollinger para una serie de precios.
    
    Args:
        prices (pd.Series): Serie de precios de cierre.
        window (int, optional): Período para la media móvil. Default: 20.
        num_std (int, optional): Número de desviaciones estándar. Default: 2.
        
    Returns:
        tuple: (upper_band, middle_band, lower_band) como pandas Series.
    """
    middle_band = prices.rolling(window=window).mean()
    std_dev = prices.rolling(window=window).std()
    
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)
    
    return upper_band, middle_band, lower_band
```

### Jupyter Notebooks

- Usa nombres descriptivos para los notebooks
- Incluye explicaciones claras en las celdas markdown
- Divide el análisis en secciones lógicas
- Limpia las salidas antes de hacer commit

## Pruebas

Para garantizar la robustez del código:

1. **Añade tests unitarios** para nuevas funcionalidades:
   ```python
   def test_bollinger_bands():
       # Arrange
       prices = pd.Series([100, 105, 110, 115, 110, 105, 100, 95, 90, 95,
                           100, 105, 110, 115, 120, 125, 130, 135, 140, 145])
       # Act
       upper, middle, lower = calculate_bollinger_bands(prices)
       # Assert
       assert len(upper) == len(prices)
       assert all(upper >= middle)
       assert all(middle >= lower)
   ```

2. **Ejecuta los tests** antes de enviar un Pull Request:
   ```bash
   pytest
   ```

## Documentación

1. **Mantén la documentación actualizada** con tus cambios
2. **Documenta decisiones técnicas importantes** 
3. **Actualiza el README.md** si es necesario
4. **Incluye ejemplos de uso** para nuevas características

### Ejemplo de Documentación para una Nueva Característica

```markdown
## Nuevo Indicador: Adaptative RSI

El RSI Adaptativo ajusta dinámicamente su período basado en la volatilidad del mercado.

### Implementación

```python
from src.features import adaptive_rsi

# Calcular RSI adaptativo
a_rsi = adaptive_rsi(data['close'], min_period=7, max_period=21)
```

### Parámetros

- `min_period`: Período mínimo en condiciones de alta volatilidad
- `max_period`: Período máximo en condiciones de baja volatilidad

### Ejemplo de Uso

```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(data.index, a_rsi)
plt.axhline(y=70, color='r', linestyle='-')
plt.axhline(y=30, color='g', linestyle='-')
plt.title('Adaptive RSI')
plt.show()
```
```

---

¡Gracias por contribuir al proyecto! Tus aportes son fundamentales para mejorar la calidad y utilidad de nuestro sistema de análisis de Bitcoin.
