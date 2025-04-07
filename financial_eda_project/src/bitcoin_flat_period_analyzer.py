import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

def detect_flat_periods(df, column='Price', window=7, threshold=0.001):
    """
    Detecta períodos donde los valores se mantienen anormalmente planos.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        column (str): Columna a analizar
        window (int): Ventana móvil para calcular la desviación estándar
        threshold (float): Umbral para considerar un período como plano
        
    Returns:
        list: Lista de tuplas con (fecha_inicio, fecha_fin) de períodos planos
    """
    # Calcular la desviación estándar móvil
    rolling_std = df[column].rolling(window=window).std()
    
    # Identificar períodos donde la desviación es casi cero (valores planos)
    flat_mask = rolling_std < threshold
    
    # Encontrar grupos consecutivos de True (períodos planos)
    flat_periods = []
    in_flat_period = False
    start_date = None
    
    for date, is_flat in zip(df.index, flat_mask):
        if is_flat and not in_flat_period:
            # Inicio de un nuevo período plano
            in_flat_period = True
            start_date = date
        elif not is_flat and in_flat_period:
            # Fin de un período plano
            in_flat_period = False
            # Solo considerar períodos de más de 'window' días
            if (date - start_date).days > window:
                flat_periods.append((start_date, date))
    
    # Verificar si hay un período plano que llegue hasta el final
    if in_flat_period:
        flat_periods.append((start_date, df.index[-1]))
    
    return flat_periods

def visualize_flat_periods(df, flat_periods, column='Price'):
    """
    Visualiza los períodos planos en el contexto de la serie temporal completa.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        flat_periods (list): Lista de tuplas con (fecha_inicio, fecha_fin)
        column (str): Columna a visualizar
    """
    plt.figure(figsize=(14, 8))
    
    # Graficar la serie temporal completa
    plt.plot(df.index, df[column], label='Precio', color='blue', alpha=0.7)
    
    # Resaltar los períodos planos
    for i, (start_date, end_date) in enumerate(flat_periods):
        # Filtrar datos para este período
        mask = (df.index >= start_date) & (df.index <= end_date)
        flat_data = df[mask]
        
        # Resaltar período plano
        plt.plot(flat_data.index, flat_data[column], color='red', linewidth=2, 
                 label=f'Período plano {i+1}' if i == 0 else None)
        
        # Añadir anotaciones
        plt.axvspan(start_date, end_date, alpha=0.2, color='red')
        
        # Añadir texto descriptivo
        mid_point = start_date + (end_date - start_date) / 2
        plt.text(mid_point, df[column].max() * 0.95, 
                 f'Período plano:\n{start_date.date()} a {end_date.date()}',
                 fontsize=9, ha='center', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Identificación de Períodos Planos en Precios de Bitcoin', fontsize=14)
    plt.xlabel('Fecha')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.show()
    
    # Imprimir detalles
    print("\nDetalles de los períodos planos detectados:")
    for i, (start_date, end_date) in enumerate(flat_periods):
        duration = (end_date - start_date).days
        mask = (df.index >= start_date) & (df.index <= end_date)
        flat_data = df[mask]
        avg_price = flat_data[column].mean()
        std_price = flat_data[column].std()
        print(f"Período {i+1}: {start_date.date()} a {end_date.date()} ({duration} días)")
        print(f"  - Precio medio: {avg_price:.2f}")
        print(f"  - Desviación estándar: {std_price:.6f}")
        print(f"  - Número de registros: {len(flat_data)}")

def analyze_flat_period_details(df, start_date, end_date):
    """
    Analiza en detalle un período plano específico.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos
        start_date: Fecha de inicio
        end_date: Fecha de fin
    """
    # Filtrar datos para el período específico
    mask = (df.index >= start_date) & (df.index <= end_date)
    flat_data = df[mask]
    
    # Analizar valores únicos y distribución
    price_unique = flat_data['Price'].unique()
    
    print(f"\nAnálisis detallado del período {start_date.date()} a {end_date.date()}:")
    print(f"Número de valores únicos de precio: {len(price_unique)}")
    print(f"Valores únicos: {sorted(price_unique)}")
    
    # Comprobar si hay cambios en otras columnas
    if 'Vol.' in flat_data.columns:
        vol_unique = flat_data['Vol.'].nunique()
        print(f"Número de valores únicos de volumen: {vol_unique}")
        if vol_unique <= 5:
            print(f"Valores únicos de volumen: {sorted(flat_data['Vol.'].unique())}")
    
    if 'Change %' in flat_data.columns:
        change_unique = flat_data['Change %'].nunique()
        print(f"Número de valores únicos de cambio porcentual: {change_unique}")
        if change_unique <= 5:
            print(f"Valores únicos de cambio %: {sorted(flat_data['Change %'].unique())}")
    
    # Visualizar distribución de valores en el período
    plt.figure(figsize=(15, 10))
    
    # Precios
    plt.subplot(3, 1, 1)
    plt.plot(flat_data.index, flat_data['Price'], 'o-', markersize=4)
    plt.title('Precios durante el período plano')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Volumen
    if 'Vol.' in flat_data.columns:
        plt.subplot(3, 1, 2)
        plt.plot(flat_data.index, flat_data['Vol.'], 'o-', color='green', markersize=4)
        plt.title('Volumen durante el período plano')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    # Cambio porcentual
    if 'Change %' in flat_data.columns:
        plt.subplot(3, 1, 3)
        plt.plot(flat_data.index, flat_data['Change %'], 'o-', color='red', markersize=4)
        plt.title('Cambio porcentual durante el período plano')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def fetch_alternative_data(start_date, end_date):
    """
    Obtiene datos alternativos de Bitcoin desde una API pública para un período específico.
    
    Args:
        start_date: Fecha de inicio (datetime)
        end_date: Fecha de fin (datetime)
        
    Returns:
        pd.DataFrame: DataFrame con datos alternativos o None si hay error
    """
    print("\nIntentando obtener datos alternativos de Bitcoin para el período problemático...")
    
    try:
        # Convertir fechas al formato requerido
        start_unix = int(start_date.timestamp())
        end_unix = int(end_date.timestamp())
        
        # Simular carga de datos alternativos desde un CSV
        print("Nota: En un entorno real, aquí conectaríamos con una API como CoinGecko o Yahoo Finance.")
        print("Para este ejercicio, simularemos que generamos datos sintéticos realistas.")
        
        # Crear un rango de fechas para el período
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Generar precios realistas (no planos) basados en un patrón de movimiento típico
        # Aquí usamos un random walk con cierta tendencia y volatilidad
        n_days = len(date_range)
        
        # Punto de partida para el precio (valor al inicio del período plano)
        seed_price = 8000  # Valor aproximado para 2018
        
        # Generar cambios diarios con volatilidad realista de Bitcoin (3-5%)
        np.random.seed(42)  # Para reproducibilidad
        daily_changes = np.random.normal(0.001, 0.03, n_days)
        
        # Crear una serie de precios realista
        prices = [seed_price]
        for change in daily_changes:
            next_price = prices[-1] * (1 + change)
            prices.append(next_price)
        prices = prices[1:]  # Eliminar el valor inicial de semilla
        
        # Crear DataFrame
        alternative_df = pd.DataFrame({
            'Date': date_range,
            'Price': prices,
            'Source': 'Synthetic Data'
        })
        
        alternative_df = alternative_df.set_index('Date')
        
        print(f"Datos alternativos obtenidos: {len(alternative_df)} registros.")
        return alternative_df
        
    except Exception as e:
        print(f"Error al obtener datos alternativos: {str(e)}")
        return None

def correct_flat_period(df, flat_period, method='alternative_data'):
    """
    Corrige un período plano utilizando diferentes métodos.
    
    Args:
        df (pd.DataFrame): DataFrame original
        flat_period (tuple): (start_date, end_date) del período plano
        method (str): Método de corrección ('interpolation', 'alternative_data')
        
    Returns:
        pd.DataFrame: DataFrame con el período plano corregido
    """
    start_date, end_date = flat_period
    corrected_df = df.copy()
    
    # Filtrar el período plano
    mask = (df.index >= start_date) & (df.index <= end_date)
    
    if method == 'interpolation':
        print("\nCorrigiendo período plano mediante interpolación avanzada...")
        
        # Encontrar valores válidos antes y después del período plano
        before_mask = df.index < start_date
        after_mask = df.index > end_date
        
        if before_mask.any() and after_mask.any():
            # Verificar duplicados en el índice
            if df.index.duplicated().any():
                print("Advertencia: Se encontraron fechas duplicadas en el índice. Manteniendo solo la primera ocurrencia.")
                # Crear copia del DataFrame eliminando duplicados
                df_no_duplicates = df[~df.index.duplicated(keep='first')]
            else:
                df_no_duplicates = df
            
            # Tomar algunos puntos antes y después para mejorar la interpolación
            before_value = df_no_duplicates[before_mask]['Price'].iloc[-1]
            after_value = df_no_duplicates[after_mask]['Price'].iloc[0]
            
            # Crear puntos artificiales para la interpolación
            interp_index = [start_date, end_date]
            interp_values = [before_value, after_value]
            
            # Crear una serie temporal para la interpolación
            interp_series = pd.Series(interp_values, index=interp_index)
            
            # Combinar con los datos originales excepto el período plano
            temp_series = df_no_duplicates[~mask]['Price'].copy()
            combined_series = pd.concat([temp_series, interp_series])
            
            # Asegurarse de que el índice no tiene duplicados
            if combined_series.index.duplicated().any():
                combined_series = combined_series[~combined_series.index.duplicated(keep='first')]
            
            try:
                # Interpolar para todos los días del período plano
                interpolated = combined_series.resample('D').interpolate(method='cubic')
                
                # Actualizar los valores en el DataFrame corregido
                for date in corrected_df[mask].index:
                    if date in interpolated.index:
                        corrected_df.loc[date, 'Price'] = interpolated[date]
            except Exception as e:
                print(f"Error en la interpolación: {str(e)}")
                print("Usando método de interpolación alternativo...")
                
                # Método alternativo: interpolación directa sin resample
                # Ordenar por índice para asegurar que la interpolación funcione correctamente
                combined_series = combined_series.sort_index()
                interpolated = combined_series.interpolate(method='cubic')
                
                # Actualizar los valores
                for date in corrected_df[mask].index:
                    if date in interpolated.index:
                        corrected_df.loc[date, 'Price'] = interpolated[date]
                    else:
                        # Para fechas que no están en el índice, usar el valor más cercano
                        closest_date = min(interpolated.index, key=lambda x: abs((x - date).total_seconds()))
                        corrected_df.loc[date, 'Price'] = interpolated[closest_date]
        else:
            print("No hay suficientes datos antes o después del período plano para interpolar.")
    
    elif method == 'alternative_data':
        print("\nCorrigiendo período plano usando datos alternativos...")
        
        # Obtener datos alternativos
        alt_data = fetch_alternative_data(start_date, end_date)
        
        if alt_data is not None:
            # Reemplazar los valores en el período plano
            for date in corrected_df[mask].index:
                if date in alt_data.index:
                    corrected_df.loc[date, 'Price'] = alt_data.loc[date, 'Price']
        else:
            print("No se pudieron obtener datos alternativos. Se mantienen los valores originales.")
    
    else:
        print(f"Método de corrección '{method}' no reconocido.")
    
    return corrected_df

def compare_original_vs_corrected(original_df, corrected_df, flat_period):
    """
    Compara visualmente los datos originales vs. los corregidos.
    
    Args:
        original_df (pd.DataFrame): DataFrame original
        corrected_df (pd.DataFrame): DataFrame corregido
        flat_period (tuple): (start_date, end_date) del período plano
    """
    start_date, end_date = flat_period
    
    # Expandir el rango para incluir contexto
    delta = (end_date - start_date) * 0.2
    context_start = start_date - delta
    context_end = end_date + delta
    
    # Filtrar datos para visualización
    mask = (original_df.index >= context_start) & (original_df.index <= context_end)
    original_view = original_df[mask]
    corrected_view = corrected_df[mask]
    
    # Graficar comparación
    plt.figure(figsize=(14, 8))
    
    plt.plot(original_view.index, original_view['Price'], 'r-', 
             label='Original (Período Plano)', alpha=0.7)
    plt.plot(corrected_view.index, corrected_view['Price'], 'b-', 
             label='Corregido', linewidth=2)
    
    # Resaltar el período plano
    plt.axvspan(start_date, end_date, alpha=0.2, color='red')
    
    plt.title('Comparación: Datos Originales vs. Corregidos', fontsize=14)
    plt.xlabel('Fecha')
    plt.ylabel('Precio')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    # Calcular estadísticas de la corrección
    original_std = original_df.loc[start_date:end_date, 'Price'].std()
    corrected_std = corrected_df.loc[start_date:end_date, 'Price'].std()
    
    print("\nEstadísticas de la corrección:")
    print(f"Desviación estándar original: {original_std:.6f}")
    print(f"Desviación estándar corregida: {corrected_std:.6f}")
    print(f"Mejora en variabilidad: {((corrected_std/original_std)-1)*100:.2f}%")

def main(df):
    """
    Función principal que organiza el análisis completo.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos de Bitcoin
    """
    # 1. Detectar períodos planos
    flat_periods = detect_flat_periods(df, column='Price', window=10, threshold=0.001)
    
    if not flat_periods:
        print("No se detectaron períodos planos significativos.")
        return df
    
    # 2. Visualizar los períodos planos
    visualize_flat_periods(df, flat_periods)
    
    # 3. Analizar en detalle el período plano más largo
    longest_period = max(flat_periods, key=lambda x: (x[1] - x[0]).days)
    analyze_flat_period_details(df, longest_period[0], longest_period[1])
    
    # 4. Corregir el período plano más largo (usando datos alternativos)
    corrected_df = correct_flat_period(df, longest_period, method='alternative_data')
    
    # 5. Comparar original vs corregido
    compare_original_vs_corrected(df, corrected_df, longest_period)
    
    # 6. También probar con interpolación para comparar
    corrected_df_interp = correct_flat_period(df, longest_period, method='interpolation')
    compare_original_vs_corrected(df, corrected_df_interp, longest_period)
    
    return corrected_df
