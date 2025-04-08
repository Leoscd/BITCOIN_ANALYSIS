# Catálogo de Características (Features)

Este documento proporciona una descripción detallada de todas las características utilizadas en el sistema de análisis y predicción de Bitcoin.

## Índice
- [Características Originales](#características-originales)
- [Indicadores Técnicos](#indicadores-técnicos)
- [Indicadores de Volatilidad](#indicadores-de-volatilidad)
- [Indicadores de Momentum](#indicadores-de-momentum)
- [Indicadores de Volumen](#indicadores-de-volumen)
- [Características Temporales](#características-temporales)
- [Características de Blockchain](#características-de-blockchain)
- [Transformaciones y Derivadas](#transformaciones-y-derivadas)

## Características Originales

Estas son las variables base obtenidas directamente de la fuente de datos:

| Característica | Descripción | Tipo |
|----------------|-------------|------|
| `timestamp` | Fecha y hora de la observación | datetime |
| `open` | Precio de apertura del período | float |
| `high` | Precio máximo del período | float |
| `low` | Precio mínimo del período | float |
| `close` | Precio de cierre del período | float |
| `volume` | Volumen de transacciones en el período | float |
| `transactions` | Número de transacciones en la blockchain | integer |
| `hashrate` | Tasa de hash de la red Bitcoin | float |
| `difficulty` | Dificultad de minado | float |

## Indicadores Técnicos

### Medias Móviles

| Característica | Descripción | Fórmula |
|----------------|-------------|---------|
| `sma_5` | Media móvil simple de 5 días | Promedio de precios de cierre de 5 días |
| `sma_10` | Media móvil simple de 10 días | Promedio de precios de cierre de 10 días |
| `sma_20` | Media móvil simple de 20 días | Promedio de precios de cierre de 20 días |
| `sma_50` | Media móvil simple de 50 días | Promedio de precios de cierre de 50 días |
| `sma_200` | Media móvil simple de 200 días | Promedio de precios de cierre de 200 días |
| `ema_5` | Media móvil exponencial de 5 días | Media ponderada exponencialmente |
| `ema_10` | Media móvil exponencial de 10 días | Media ponderada exponencialmente |
| `ema_20` | Media móvil exponencial de 20 días | Media ponderada exponencialmente |
| `ema_50` | Media móvil exponencial de 50 días | Media ponderada exponencialmente |

### Cruce de Medias Móviles

| Característica | Descripción | Fórmula |
|----------------|-------------|---------|
| `sma_5_10_cross` | Cruce de SMA 5 y 10 | 1 si SMA 5 cruza por encima de SMA 10, -1 si cruza por debajo, 0 en otro caso |
| `sma_10_20_cross` | Cruce de SMA 10 y 20 | 1 si SMA 10 cruza por encima de SMA 20, -1 si cruza por debajo, 0 en otro caso |
| `sma_50_200_cross` | Cruce de SMA 50 y 200 (Golden/Death Cross) | 1 si SMA 50 cruza por encima de SMA 200, -1 si cruza por debajo, 0 en otro caso |

### Comparativas de Precios

| Característica | Descripción | Fórmula |
|----------------|-------------|---------|
| `price_sma_5_ratio` | Ratio entre precio actual y SMA 5 | close / sma_5 |
| `price_sma_10_ratio` | Ratio entre precio actual y SMA 10 | close / sma_10 |
| `price_sma_20_ratio` | Ratio entre precio actual y SMA 20 | close / sma_20 |
| `price_sma_50_ratio` | Ratio entre precio actual y SMA 50 | close / sma_50 |
| `price_sma_200_ratio` | Ratio entre precio actual y SMA 200 | close / sma_200 |

### MACD (Moving Average Convergence Divergence)

| Característica | Descripción | Fórmula |
|----------------|-------------|---------|
| `macd_line` | Línea MACD | EMA 12 - EMA 26 |
| `macd_signal` | Línea de señal MACD | EMA 9 de la línea MACD |
| `macd_histogram` | Histograma MACD | macd_line - macd_signal |
| `macd_cross` | Cruce de MACD | 1 si MACD cruza por encima de la señal, -1 si cruza por debajo, 0 en otro caso |

## Indicadores de Volatilidad

### Bandas de Bollinger

| Característica | Descripción | Fórmula |
|----------------|-------------|---------|
| `bollinger_upper` | Banda superior de Bollinger | SMA 20 + 2 * desviación estándar de 20 días |
| `bollinger_middle` | Banda media de Bollinger | SMA 20 |
| `bollinger_lower` | Banda inferior de Bollinger | SMA 20 - 2 * desviación estándar de 20 días |
| `bollinger_width` | Ancho de las bandas de Bollinger | (bollinger_upper - bollinger_lower) / bollinger_middle |
| `bollinger_position` | Posición relativa dentro de las bandas | (close - bollinger_lower) / (bollinger_upper - bollinger_lower) |

### Medidas de Volatilidad

| Característica | Descripción | Fórmula |
|----------------|-------------|---------|
| `daily_range` | Rango diario | (high - low) / open |
| `daily_return` | Retorno diario | (close - open) / open |
| `atr_14` | Average True Range (14 días) | Media móvil de los True Range de 14 días |
| `volatility_5d` | Volatilidad de 5 días | Desviación estándar de retornos diarios en 5 días |
| `volatility_10d` | Volatilidad de 10 días | Desviación estándar de retornos diarios en 10 días |
| `volatility_20d` | Volatilidad de 20 días | Desviación estándar de retornos diarios en 20 días |
| `volatility_ratio_5_20` | Ratio de volatilidad | volatility_5d / volatility_20d |

## Indicadores de Momentum

### RSI (Relative Strength Index)

| Característica | Descripción | Fórmula |
|----------------|-------------|---------|
| `rsi_14` | RSI de 14 días | 100 - (100 / (1 + RS)), donde RS = promedio de ganancias / promedio de pérdidas |
| `rsi_7` | RSI de 7 días | 100 - (100 / (1 + RS)), donde RS = promedio de ganancias / promedio de pérdidas |
| `rsi_21` | RSI de 21 días | 100 - (100 / (1 + RS)), donde RS = promedio de ganancias / promedio de pérdidas |
| `rsi_divergence` | Divergencia de RSI | 1 si precio sube y RSI baja, -1 si precio baja y RSI sube, 0 en otro caso |

### Estocástico

| Característica | Descripción | Fórmula |
|----------------|-------------|---------|
| `stochastic_k` | %K Estocástico (14 días) | ((close - min(low, 14)) / (max(high, 14) - min(low, 14))) * 100 |
| `stochastic_d` | %D Estocástico | Media móvil de 3 días de %K |
| `stochastic_cross` | Cruce estocástico | 1 si %K cruza por encima de %D, -1 si cruza por debajo, 0 en otro caso |

### Otros Indicadores de Momentum

| Característica | Descripción | Fórmula |
|----------------|-------------|---------|
| `roc_10` | Rate of Change (10 días) | ((close - close[10]) / close[10]) * 100 |
| `cci_20` | Commodity Channel Index (20 días) | (precio típico - SMA del precio típico) / (0.015 * desviación media) |
| `adx_14` | Average Directional Index (14 días) | Media móvil del DX |
| `williams_r` | Williams %R (14 días) | ((max(high, 14) - close) / (max(high, 14) - min(low, 14))) * -100 |

## Indicadores de Volumen

| Característica | Descripción | Fórmula |
|----------------|-------------|---------|
| `volume_sma_5` | Media móvil de volumen (5 días) | Promedio de volumen de 5 días |
| `volume_sma_10` | Media móvil de volumen (10 días) | Promedio de volumen de 10 días |
| `volume_ratio_5_10` | Ratio de volumen | volume_sma_5 / volume_sma_10 |
| `obv` | On-Balance Volume | Suma acumulativa de volumen (positivo si precio sube, negativo si baja) |
| `volume_price_trend` | Volume Price Trend | Volumen * (close - close[1]) / close[1] |
| `volume_breakout` | Breakout de volumen | 1 si volumen > 2 * volume_sma_10, 0 en otro caso |

## Características Temporales

| Característica | Descripción | Tipo |
|----------------|-------------|------|
| `day_of_week` | Día de la semana | categorical (0-6) |
| `hour_of_day` | Hora del día | categorical (0-23) |
| `is_weekend` | Indicador de fin de semana | binary (0/1) |
| `is_month_end` | Indicador de fin de mes | binary (0/1) |
| `is_quarter_end` | Indicador de fin de trimestre | binary (0/1) |
| `month` | Mes del año | categorical (1-12) |

## Características de Blockchain

| Característica | Descripción | Fórmula |
|----------------|-------------|---------|
| `transactions_sma_7` | Media móvil de transacciones (7 días) | Promedio de transacciones de 7 días |
| `transactions_growth` | Crecimiento de transacciones | (transactions - transactions[7]) / transactions[7] |
| `hashrate_growth` | Crecimiento de hashrate | (hashrate - hashrate[7]) / hashrate[7] |
| `difficulty_change` | Cambio en dificultad | (difficulty - difficulty[14]) / difficulty[14] |
| `transaction_value` | Valor promedio de transacción | Estimado a partir de volume / transactions |

## Transformaciones y Derivadas

| Característica | Descripción | Fórmula |
|----------------|-------------|---------|
| `log_return` | Retorno logarítmico | ln(close / close[1]) |
| `cum_return_5d` | Retorno acumulado (5 días) | (close / close[5]) - 1 |
| `cum_return_10d` | Retorno acumulado (10 días) | (close / close[10]) - 1 |
| `cum_return_20d` | Retorno acumulado (20 días) | (close / close[20]) - 1 |
| `z_score_20d` | Z-score del precio (20 días) | (close - SMA 20) / desviación estándar de 20 días |
| `price_acceleration` | Aceleración del precio | (close - 2*close[1] + close[2]) / close[2] |
| `momentum_5d` | Momentum de 5 días | close - close[5] |
| `momentum_10d` | Momentum de 10 días | close - close[10] |

## Variables Objetivo

| Característica | Descripción | Tipo |
|----------------|-------------|------|
| `target_movement` | Movimiento significativo en 24h (±3%) | binary (0/1) |
| `target_return` | Retorno porcentual en 24h | float |
| `target_direction` | Dirección del movimiento en 24h | categorical (-1/0/1) |

## Notas sobre Feature Engineering

- Las características se calcularon utilizando la biblioteca `ta` de Python para indicadores técnicos estándar
- Se implementaron funciones personalizadas para indicadores más complejos o específicos
- Se aplicó normalización y escalado a las características según fuera necesario
- Se realizó análisis de correlación para evitar multicolinealidad
- Las 20-25 características más importantes se seleccionaron mediante análisis de importancia de Random Forest
