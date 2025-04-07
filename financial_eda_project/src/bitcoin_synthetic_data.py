"""
Módulos para Corrección de Período Plano en Bitcoin (2010-07-18 a 2010-10-16)
mediante generación sintética de datos que preserven propiedades estadísticas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --------------------------------
# MÓDULO DE PREPARACIÓN DE DATOS
# --------------------------------

def prepare_data(df, start_date='2010-07-18', end_date='2010-10-16', 
                adjacent_days=30):
    """
    Prepara los datos para el análisis, identificando el período plano y 
    extrayendo períodos adyacentes para análisis.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos históricos de Bitcoin (con 'Date' como índice)
    start_date : str
        Fecha de inicio del período plano en formato 'YYYY-MM-DD'
    end_date : str
        Fecha de fin del período plano en formato 'YYYY-MM-DD'
    adjacent_days : int
        Número de días a considerar antes y después del período plano
        
    Retorna:
    --------
    dict
        Diccionario con los siguientes DataFrames:
        - 'df_flat': datos durante el período plano
        - 'df_before': datos del período anterior al plano
        - 'df_after': datos del período posterior al plano
        - 'flat_period_info': información sobre el período plano (duración, etc.)
    """
    try:
        # Convertir fechas a datetime si son strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Verificar que el índice sea datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Advertencia: El índice no es datetime. Intentando convertir...")
            df.index = pd.to_datetime(df.index)
        
        # Ordenar el DataFrame por fecha (ascendente)
        df = df.sort_index()
        
        # Extraer el período plano
        mask_flat = (df.index >= start_date) & (df.index <= end_date)
        df_flat = df[mask_flat].copy()
        
        # Extraer períodos adyacentes
        before_start = start_date - pd.Timedelta(days=adjacent_days)
        after_end = end_date + pd.Timedelta(days=adjacent_days)
        
        mask_before = (df.index >= before_start) & (df.index < start_date)
        mask_after = (df.index > end_date) & (df.index <= after_end)
        
        df_before = df[mask_before].copy()
        df_after = df[mask_after].copy()
        
        # Calcular información sobre el período plano
        days_in_period = (end_date - start_date).days + 1
        
        flat_period_info = {
            'start_date': start_date,
            'end_date': end_date,
            'days_in_period': days_in_period,
            'num_records': len(df_flat)
        }
        
        print(f"Período plano: {start_date.date()} a {end_date.date()} ({days_in_period} días)")
        print(f"Registros en período plano: {len(df_flat)}")
        print(f"Registros en período anterior: {len(df_before)}")
        print(f"Registros en período posterior: {len(df_after)}")
        
        # Verificar si hay suficientes datos en períodos adyacentes
        if len(df_before) < 5 or len(df_after) < 5:
            print("Advertencia: Pocos datos en períodos adyacentes. Esto puede afectar el análisis.")
        
        return {
            'df_flat': df_flat,
            'df_before': df_before,
            'df_after': df_after,
            'flat_period_info': flat_period_info
        }
    
    except Exception as e:
        print(f"Error al preparar los datos: {str(e)}")
        return None


# ---------------------------------
# MÓDULO DE ANÁLISIS ESTADÍSTICO
# ---------------------------------

def analyze_statistics(data_dict):
    """
    Analiza estadísticamente los períodos antes, durante y después del período plano.
    
    Parámetros:
    -----------
    data_dict : dict
        Diccionario con DataFrames según lo retornado por prepare_data()
        
    Retorna:
    --------
    dict
        Diccionario con estadísticas y propiedades de los datos
    """
    try:
        df_flat = data_dict['df_flat']
        df_before = data_dict['df_before']
        df_after = data_dict['df_after']
        
        # Combinar períodos adyacentes para análisis
        df_adjacent = pd.concat([df_before, df_after])
        
        # Estadísticas para precio
        price_stats = {}
        for period_name, df in [('before', df_before), ('flat', df_flat), ('after', df_after), 
                               ('adjacent', df_adjacent)]:
            if 'Price' in df.columns:
                price_col = 'Price'
            elif 'Close' in df.columns:
                price_col = 'Close'
            else:
                # Buscar primera columna numérica que podría ser precio
                price_col = df.select_dtypes(include=[np.number]).columns[0]
                print(f"Advertencia: Usando '{price_col}' como columna de precio.")
            
            # Estadísticas básicas
            stats_dict = {
                'mean': df[price_col].mean(),
                'median': df[price_col].median(),
                'std': df[price_col].std(),
                'min': df[price_col].min(),
                'max': df[price_col].max(),
                'skew': df[price_col].skew(),
                'kurtosis': df[price_col].kurtosis()
            }
            
            # Calcular retornos logarítmicos diarios si hay suficientes datos
            if len(df) > 1:
                log_returns = np.log(df[price_col] / df[price_col].shift(1)).dropna()
                
                # Volatilidad diaria (desviación estándar de retornos log)
                stats_dict['daily_volatility'] = log_returns.std()
                
                # Tendencia (pendiente de la regresión lineal)
                if len(df) > 2:
                    x = np.arange(len(df))
                    y = df[price_col].values
                    slope, _, _, _, _ = stats.linregress(x, y)
                    stats_dict['trend_slope'] = slope
                    
                    # Autocorrelación en retornos (lag 1)
                    if len(log_returns) > 2:
                        stats_dict['autocorrelation'] = log_returns.autocorr(1)
            
            price_stats[period_name] = stats_dict
        
        # Analizar volumen si está disponible
        volume_stats = {}
        if 'Vol.' in df_adjacent.columns or 'Volume' in df_adjacent.columns:
            vol_col = 'Vol.' if 'Vol.' in df_adjacent.columns else 'Volume'
            
            for period_name, df in [('before', df_before), ('flat', df_flat), ('after', df_after), 
                                  ('adjacent', df_adjacent)]:
                if vol_col in df.columns:
                    volume_stats[period_name] = {
                        'mean': df[vol_col].mean(),
                        'median': df[vol_col].median(),
                        'std': df[vol_col].std()
                    }
        
        # Analizar cambio porcentual si está disponible
        change_stats = {}
        if 'Change %' in df_adjacent.columns:
            for period_name, df in [('before', df_before), ('flat', df_flat), ('after', df_after), 
                                  ('adjacent', df_adjacent)]:
                if 'Change %' in df.columns:
                    change_stats[period_name] = {
                        'mean': df['Change %'].mean(),
                        'median': df['Change %'].median(),
                        'std': df['Change %'].std()
                    }
        
        # Identificar propiedades de la distribución log-normal
        log_normal_params = {}
        if len(df_adjacent) > 1:
            price_col = 'Price' if 'Price' in df_adjacent.columns else 'Close'
            
            # Comprobar si los precios siguen una distribución log-normal
            log_prices = np.log(df_adjacent[price_col])
            shapiro_test = stats.shapiro(log_prices)
            
            # Parámetros de la distribución log-normal
            shape, loc, scale = stats.lognorm.fit(df_adjacent[price_col])
            
            log_normal_params = {
                'shapiro_test_stat': shapiro_test[0],
                'shapiro_test_p': shapiro_test[1],
                'is_lognormal': shapiro_test[1] > 0.05,  # p > 0.05 sugiere distribución normal para log-precios
                'lognorm_shape': shape,
                'lognorm_loc': loc,
                'lognorm_scale': scale
            }
        
        # Resultados del análisis
        analysis_results = {
            'price_stats': price_stats,
            'volume_stats': volume_stats,
            'change_stats': change_stats,
            'log_normal_params': log_normal_params
        }
        
        print("\nResumen de estadísticas de precios:")
        for period, stats in price_stats.items():
            print(f"\n{period.capitalize()}:")
            print(f"  Media: {stats['mean']:.4f}")
            print(f"  Mediana: {stats['median']:.4f}")
            print(f"  Desv. estándar: {stats['std']:.4f}")
            if 'daily_volatility' in stats:
                print(f"  Volatilidad diaria: {stats['daily_volatility']:.4f}")
            if 'trend_slope' in stats:
                print(f"  Tendencia (pendiente): {stats['trend_slope']:.6f}")
        
        return analysis_results
    
    except Exception as e:
        print(f"Error en el análisis estadístico: {str(e)}")
        return None


# --------------------------------------
# MÓDULO DE GENERACIÓN SINTÉTICA
# --------------------------------------

def generate_synthetic_data_arima(data_dict, analysis_results):
    """
    Genera datos sintéticos para el período plano usando modelo ARIMA
    calibrado con datos de períodos adyacentes.
    
    Parámetros:
    -----------
    data_dict : dict
        Diccionario con DataFrames según lo retornado por prepare_data()
    analysis_results : dict
        Diccionario con resultados del análisis estadístico
        
    Retorna:
    --------
    pandas.DataFrame
        DataFrame con los datos sintéticos generados para el período plano
    """
    try:
        df_flat = data_dict['df_flat']
        df_before = data_dict['df_before']
        df_after = data_dict['df_after']
        flat_period_info = data_dict['flat_period_info']
        
        # Combinamos datos adyacentes para estimar el modelo
        df_adjacent = pd.concat([df_before, df_after])
        
        # Determinamos la columna de precio
        if 'Price' in df_adjacent.columns:
            price_col = 'Price'
        elif 'Close' in df_adjacent.columns:
            price_col = 'Close'
        else:
            price_col = df_adjacent.select_dtypes(include=[np.number]).columns[0]
        
        # Trabajamos con log-precios para mantener la positividad
        log_prices = np.log(df_adjacent[price_col])
        
        # Identificar orden óptimo del modelo ARIMA
        # Usamos AIC para seleccionar automáticamente el mejor modelo
        best_aic = float('inf')
        best_order = None
        
        # Probamos diferentes combinaciones de órdenes ARIMA
        for p in range(6):  # AR
            for d in range(2):  # I
                for q in range(6):  # MA
                    # Limitamos a modelos simples para evitar sobreajuste
                    if p + d + q <= 5:
                        try:
                            model = ARIMA(log_prices, order=(p, d, q))
                            results = model.fit()
                            aic = results.aic
                            
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                        except:
                            continue
        
        print(f"\nMejor orden ARIMA: {best_order} (AIC: {best_aic:.4f})")
        
        # Ajustar el modelo ARIMA con el mejor orden
        model = ARIMA(log_prices, order=best_order)
        results = model.fit()
        
        # Crear índice para el período plano
        if len(df_flat) > 0:
            synth_index = df_flat.index
        else:
            # Crear índice si no hay datos en el período plano
            synth_index = pd.date_range(
                start=flat_period_info['start_date'],
                end=flat_period_info['end_date'],
                freq='D'
            )
        
        # Generar predicciones para el período plano
        n_steps = len(synth_index)
        log_forecast = results.forecast(steps=n_steps)
        
        # Convertir log-precios a precios y añadir ruido para realismo
        forecast = np.exp(log_forecast)
        
        # Añadir variabilidad realista basada en la volatilidad observada
        daily_vol = analysis_results['price_stats']['adjacent']['daily_volatility']
        
        # Generar ruido multiplicativo log-normal
        np.random.seed(42)  # Para reproducibilidad
        noise = np.exp(np.random.normal(0, daily_vol, n_steps))
        forecast_with_noise = forecast * noise
        
        # Crear DataFrame con los datos sintéticos
        synthetic_df = pd.DataFrame(index=synth_index)
        synthetic_df[price_col] = forecast_with_noise
        
        # Generar volumen sintético si está disponible en los datos originales
        if 'Vol.' in df_adjacent.columns or 'Volume' in df_adjacent.columns:
            vol_col = 'Vol.' if 'Vol.' in df_adjacent.columns else 'Volume'
            
            # Estadísticas de volumen
            vol_mean = analysis_results['volume_stats']['adjacent']['mean']
            vol_std = analysis_results['volume_stats']['adjacent']['std']
            
            # Generar volumen con distribución similar
            synthetic_volume = np.random.lognormal(
                np.log(vol_mean) - 0.5 * np.log(1 + (vol_std/vol_mean)**2),
                np.sqrt(np.log(1 + (vol_std/vol_mean)**2)),
                n_steps
            )
            
            synthetic_df[vol_col] = synthetic_volume
        
        # Generar cambio porcentual si está disponible
        if 'Change %' in df_adjacent.columns:
            # Calcular cambios porcentuales diarios a partir de los precios sintéticos
            synthetic_df['Change %'] = synthetic_df[price_col].pct_change() * 100
            
            # Rellenar el primer valor con un valor plausible
            if len(synthetic_df) > 0:
                synthetic_df['Change %'].iloc[0] = np.random.choice(
                    df_adjacent['Change %'].dropna().values
                )
        
        print(f"Datos sintéticos generados con modelo ARIMA: {len(synthetic_df)} registros")
        return synthetic_df
    
    except Exception as e:
        print(f"Error al generar datos sintéticos con ARIMA: {str(e)}")
        return None


def generate_synthetic_data_bootstrap(data_dict, analysis_results, block_size=5):
    """
    Genera datos sintéticos para el período plano usando bootstrap con bloques móviles
    para preservar la autocorrelación en los datos.
    
    Parámetros:
    -----------
    data_dict : dict
        Diccionario con DataFrames según lo retornado por prepare_data()
    analysis_results : dict
        Diccionario con resultados del análisis estadístico
    block_size : int
        Tamaño de los bloques para el bootstrap (en días)
        
    Retorna:
    --------
    pandas.DataFrame
        DataFrame con los datos sintéticos generados para el período plano
    """
    try:
        df_flat = data_dict['df_flat']
        df_before = data_dict['df_before']
        df_after = data_dict['df_after']
        flat_period_info = data_dict['flat_period_info']
        
        # Combinamos datos adyacentes para el bootstrap
        df_adjacent = pd.concat([df_before, df_after])
        
        # Determinamos la columna de precio
        if 'Price' in df_adjacent.columns:
            price_col = 'Price'
        elif 'Close' in df_adjacent.columns:
            price_col = 'Close'
        else:
            price_col = df_adjacent.select_dtypes(include=[np.number]).columns[0]
        
        # Crear índice para el período plano
        if len(df_flat) > 0:
            synth_index = df_flat.index
        else:
            # Crear índice si no hay datos en el período plano
            synth_index = pd.date_range(
                start=flat_period_info['start_date'],
                end=flat_period_info['end_date'],
                freq='D'
            )
        
        n_days = len(synth_index)
        
        # Calcular retornos logarítmicos del período adyacente
        log_returns = np.log(df_adjacent[price_col] / df_adjacent[price_col].shift(1)).dropna()
        
        # Realizar bootstrap con bloques para preservar autocorrelación
        n_blocks_needed = int(np.ceil(n_days / block_size))
        
        # Disponibilidad de bloques en datos adyacentes
        max_block_start = len(log_returns) - block_size
        
        if max_block_start <= 0:
            print("Advertencia: No hay suficientes datos para bootstrap con bloques.")
            return generate_synthetic_data_arima(data_dict, analysis_results)
        
        # Seleccionar bloques aleatoriamente
        np.random.seed(42)  # Para reproducibilidad
        block_starts = np.random.randint(0, max_block_start, n_blocks_needed)
        
        # Construir serie de retornos sintéticos
        sampled_returns = []
        for start in block_starts:
            block = log_returns.iloc[start:start+block_size].values
            sampled_returns.extend(block)
        
        # Recortar al número exacto de días necesarios
        sampled_returns = sampled_returns[:n_days-1]  # -1 porque el primer día no tiene retorno
        
        # Convertir retornos a precios
        # Comenzar desde el último precio conocido antes del período plano
        if len(df_before) > 0:
            start_price = df_before[price_col].iloc[-1]
        else:
            start_price = df_after[price_col].iloc[0]
        
        # Construir la serie de precios
        synthetic_prices = [start_price]
        for ret in sampled_returns:
            next_price = synthetic_prices[-1] * np.exp(ret)
            synthetic_prices.append(next_price)
        
        # Crear DataFrame con los datos sintéticos
        synthetic_df = pd.DataFrame(index=synth_index)
        synthetic_df[price_col] = synthetic_prices
        
        # Generar volumen sintético si está disponible en los datos originales
        if 'Vol.' in df_adjacent.columns or 'Volume' in df_adjacent.columns:
            vol_col = 'Vol.' if 'Vol.' in df_adjacent.columns else 'Volume'
            
            # Bootstrap para volumen también
            volumes = df_adjacent[vol_col].values
            
            # Seleccionar bloques de volumen alineados con los precios
            sampled_volumes = []
            for start in block_starts:
                if start + block_size <= len(volumes):
                    block = volumes[start:start+block_size]
                    sampled_volumes.extend(block)
            
            # Recortar y asignar
            sampled_volumes = sampled_volumes[:n_days]
            synthetic_df[vol_col] = sampled_volumes
        
        # Generar cambio porcentual si está disponible
        if 'Change %' in df_adjacent.columns:
            # Calcular cambios porcentuales diarios a partir de los precios sintéticos
            synthetic_df['Change %'] = synthetic_df[price_col].pct_change() * 100
            
            # Rellenar el primer valor con un valor plausible
            if len(synthetic_df) > 0:
                synthetic_df['Change %'].iloc[0] = np.random.choice(
                    df_adjacent['Change %'].dropna().values
                )
        
        print(f"Datos sintéticos generados con bootstrap de bloques: {len(synthetic_df)} registros")
        print(f"Tamaño de bloque: {block_size} días, Bloques utilizados: {n_blocks_needed}")
        return synthetic_df
    
    except Exception as e:
        print(f"Error al generar datos sintéticos con bootstrap: {str(e)}")
        # Fallback a ARIMA si bootstrap falla
        print("Intentando con método ARIMA como alternativa...")
        return generate_synthetic_data_arima(data_dict, analysis_results)


# -------------------------------
# MÓDULO DE VALIDACIÓN
# -------------------------------

def validate_synthetic_data(synthetic_df, data_dict, analysis_results):
    """
    Valida la calidad de los datos sintéticos generados.
    
    Parámetros:
    -----------
    synthetic_df : pandas.DataFrame
        DataFrame con los datos sintéticos generados
    data_dict : dict
        Diccionario con DataFrames según lo retornado por prepare_data()
    analysis_results : dict
        Diccionario con resultados del análisis estadístico
        
    Retorna:
    --------
    dict
        Diccionario con métricas de validación y resultados
    """
    try:
        df_adjacent = pd.concat([data_dict['df_before'], data_dict['df_after']])
        
        # Determinamos la columna de precio
        if 'Price' in synthetic_df.columns:
            price_col = 'Price'
        elif 'Close' in synthetic_df.columns:
            price_col = 'Close'
        else:
            price_col = synthetic_df.select_dtypes(include=[np.number]).columns[0]
        
        # Calcular retornos logarítmicos
        synth_log_returns = np.log(synthetic_df[price_col] / synthetic_df[price_col].shift(1)).dropna()
        
        if len(df_adjacent) > 1:
            adj_log_returns = np.log(df_adjacent[price_col] / df_adjacent[price_col].shift(1)).dropna()
        else:
            adj_log_returns = pd.Series()
        
        # Estadísticas de los datos sintéticos
        synthetic_stats = {
            'mean': synthetic_df[price_col].mean(),
            'median': synthetic_df[price_col].median(),
            'std': synthetic_df[price_col].std(),
            'min': synthetic_df[price_col].min(),
            'max': synthetic_df[price_col].max()
        }
        
        if len(synth_log_returns) > 1:
            synthetic_stats['daily_volatility'] = synth_log_returns.std()
            
            if len(synth_log_returns) > 2:
                synthetic_stats['autocorrelation'] = synth_log_returns.autocorr(1)
                
                # Test de normalidad en retornos logarítmicos
                shapiro_test = stats.shapiro(synth_log_returns)
                synthetic_stats['shapiro_test_p'] = shapiro_test[1]
        
        # Comparación con estadísticas de períodos adyacentes
        comparison = {}
        adjacent_stats = analysis_results['price_stats']['adjacent']
        
        # Calcular diferencia porcentual para métricas clave
        for key in ['mean', 'std', 'daily_volatility']:
            if key in synthetic_stats and key in adjacent_stats:
                pct_diff = (synthetic_stats[key] - adjacent_stats[key]) / adjacent_stats[key] * 100
                comparison[f'{key}_pct_diff'] = pct_diff
        
        # Prueba de Kolmogorov-Smirnov para comparar distribuciones
        if len(synth_log_returns) > 5 and len(adj_log_returns) > 5:
            ks_stat, ks_pvalue = stats.ks_2samp(synth_log_returns, adj_log_returns)
            comparison['ks_stat'] = ks_stat
            comparison['ks_pvalue'] = ks_pvalue
            comparison['distributions_similar'] = ks_pvalue > 0.05
        
        # Evaluar continuidad en los bordes del período
        edge_continuity = {}
        
        if len(data_dict['df_before']) > 0 and len(synthetic_df) > 0:
            before_last_price = data_dict['df_before'][price_col].iloc[-1]
            synth_first_price = synthetic_df[price_col].iloc[0]
            edge_continuity['before_synth_jump_pct'] = (synth_first_price - before_last_price) / before_last_price * 100
        
        if len(data_dict['df_after']) > 0 and len(synthetic_df) > 0:
            after_first_price = data_dict['df_after'][price_col].iloc[0]
            synth_last_price = synthetic_df[price_col].iloc[-1]
            edge_continuity['synth_after_jump_pct'] = (after_first_price - synth_last_price) / synth_last_price * 100
        
        # Resultados de validación
        validation_results = {
            'synthetic_stats': synthetic_stats,
            'comparison': comparison,
            'edge_continuity': edge_continuity,
            'passed_validation': True  # Por defecto, consideramos que pasa
        }
        
        # Criterios para considerar que la validación ha fallado
        if 'distributions_similar' in comparison and not comparison['distributions_similar']:
            validation_results['passed_validation'] = False
            validation_results['failure_reason'] = "Las distribuciones de retornos no son similares"
        
        for key in ['mean_pct_diff', 'std_pct_diff', 'daily_volatility_pct_diff']:
            if key in comparison and abs(comparison[key]) > 30:  # 30% de diferencia
                validation_results['passed_validation'] = False
                validation_results['failure_reason'] = f"Diferencia excesiva en {key}"
                break
        
        for key in ['before_synth_jump_pct', 'synth_after_jump_pct']:
            if key in edge_continuity and abs(edge_continuity[key]) > 10:  # 10% de salto
                validation_results['passed_validation'] = False
                validation_results['failure_reason'] = f"Discontinuidad excesiva en {key}"
                break
        
        # Imprimir resultados de validación
        print("\nResultados de validación:")
        print(f"Media original: {adjacent_stats['mean']:.4f}, Sintética: {synthetic_stats['mean']:.4f}")
        if 'daily_volatility' in adjacent_stats and 'daily_volatility' in synthetic_stats:
            print(f"Volatilidad original: {adjacent_stats['daily_volatility']:.4f}, "
                  f"Sintética: {synthetic_stats['daily_volatility']:.4f}")
        
        for key, value in comparison.items():
            if 'pct_diff' in key:
                print(f"Diferencia en {key.split('_')[0]}: {value:.2f}%")
        
        if 'distributions_similar' in comparison:
            print(f"Distribuciones similares: {'Sí' if comparison['distributions_similar'] else 'No'} "
                  f"(p-value: {comparison['ks_pvalue']:.4f})")
        
        print(f"Validación superada: {'Sí' if validation_results['passed_validation'] else 'No'}")
        if not validation_results['passed_validation']:
            print(f"Motivo: {validation_results['failure_reason']}")
        
        return validation_results
    
    except Exception as e:
        print(f"Error en la validación de datos sintéticos: {str(e)}")
        return {'passed_validation': False, 'failure_reason': str(e)}


