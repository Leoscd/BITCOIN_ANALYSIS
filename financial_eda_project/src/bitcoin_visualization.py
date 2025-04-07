"""
Módulo de visualización para datos sintéticos de Bitcoin
Permite visualizar y comparar datos originales vs. sintéticos
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

def visualize_results(synthetic_df, data_dict, validation_results):
    """
    Visualiza los datos originales y sintéticos para comparación.
    
    Parámetros:
    -----------
    synthetic_df : pandas.DataFrame
        DataFrame con los datos sintéticos generados
    data_dict : dict
        Diccionario con DataFrames según lo retornado por prepare_data()
    validation_results : dict
        Diccionario con los resultados de la validación
    """
    try:
        # Configuración de visualización
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
        # Extraer DataFrames
        df_flat = data_dict['df_flat']
        df_before = data_dict['df_before']
        df_after = data_dict['df_after']
        flat_period_info = data_dict['flat_period_info']
        
        # Determinar columna de precio
        if 'Price' in synthetic_df.columns:
            price_col = 'Price'
        elif 'Close' in synthetic_df.columns:
            price_col = 'Close'
        else:
            price_col = synthetic_df.select_dtypes(include=[np.number]).columns[0]
        
        # 1. Gráfico principal: Serie temporal completa con datos sintéticos
        plt.figure(figsize=(15, 8))
        
        # Datos antes del período plano
        plt.plot(df_before.index, df_before[price_col], color='blue', 
                label='Datos originales (antes)', linewidth=2)
        
        # Datos planos originales (mostrados en rojo)
        plt.plot(df_flat.index, df_flat[price_col], color='red', 
                label='Período plano original', linewidth=2, alpha=0.5)
        
        # Datos sintéticos (en verde)
        plt.plot(synthetic_df.index, synthetic_df[price_col], color='green', 
                label='Datos sintéticos', linewidth=2.5)
        
        # Datos después del período plano
        plt.plot(df_after.index, df_after[price_col], color='blue', 
                label='Datos originales (después)', linewidth=2)
        
        # Resaltar período plano
        plt.axvspan(flat_period_info['start_date'], flat_period_info['end_date'], 
                   color='lightyellow', alpha=0.3, label='Período plano')
        
        # Añadir anotaciones sobre el período
        plt.title('Corrección de Período Plano en Bitcoin (2010-07-18 a 2010-10-16)', fontsize=16)
        plt.xlabel('Fecha', fontsize=14)
        plt.ylabel(f'{price_col} (USD)', fontsize=14)
        plt.legend()
        
        # Formatear eje x con fechas
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        # Añadir grid y ajustar layout
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('bitcoin_synthetic_series.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Comparación de distribuciones de retornos
        plt.figure(figsize=(15, 10))
        
        # Crear un subplot para histogramas
        plt.subplot(2, 1, 1)
        
        # Calcular retornos logarítmicos
        df_adjacent = pd.concat([df_before, df_after])
        
        log_returns_orig = np.log(df_adjacent[price_col] / df_adjacent[price_col].shift(1)).dropna()
        log_returns_synth = np.log(synthetic_df[price_col] / synthetic_df[price_col].shift(1)).dropna()
        
        # Histograma de retornos originales
        sns.histplot(log_returns_orig, color='blue', alpha=0.5, 
                    label='Retornos originales', bins=20, kde=True)
        
        # Histograma de retornos sintéticos
        sns.histplot(log_returns_synth, color='green', alpha=0.5, 
                    label='Retornos sintéticos', bins=20, kde=True)
        
        plt.title('Distribución de Retornos Logarítmicos', fontsize=14)
        plt.xlabel('Retorno Logarítmico', fontsize=12)
        plt.ylabel('Frecuencia', fontsize=12)
        plt.legend()
        
        # 3. QQ-Plot para comparar distribuciones
        plt.subplot(2, 1, 2)
        
        # Preparar datos para QQ-plot
        orig_sorted = np.sort(log_returns_orig)
        synth_sorted = np.sort(log_returns_synth)
        
        # Si los arrays tienen diferentes tamaños, interpolar para normalizar
        if len(orig_sorted) != len(synth_sorted):
            # Crear puntos equidistantes para interpolación
            orig_points = np.linspace(0, 1, len(orig_sorted))
            synth_points = np.linspace(0, 1, len(synth_sorted))
            
            # Interpolar el array más corto para que coincida con el más largo
            if len(orig_sorted) > len(synth_sorted):
                synth_interp = np.interp(orig_points, synth_points, synth_sorted)
                synth_sorted = synth_interp
            else:
                orig_interp = np.interp(synth_points, orig_points, orig_sorted)
                orig_sorted = orig_interp
        
        # Crear QQ-plot
        plt.scatter(orig_sorted, synth_sorted, alpha=0.5)
        
        # Añadir línea de referencia (y=x)
        min_val = min(orig_sorted.min(), synth_sorted.min())
        max_val = max(orig_sorted.max(), synth_sorted.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.title('QQ-Plot: Retornos Originales vs. Sintéticos', fontsize=14)
        plt.xlabel('Cuantiles de Retornos Originales', fontsize=12)
        plt.ylabel('Cuantiles de Retornos Sintéticos', fontsize=12)
        
        plt.tight_layout()
        plt.savefig('bitcoin_returns_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Gráfico de volatilidad
        plt.figure(figsize=(15, 6))
        
        # Calcular volatilidad rodante (20 días)
        window = 10  # Ventana más corta debido al período limitado
        orig_vol = log_returns_orig.rolling(window=window).std() * np.sqrt(252)  # Anualizada
        synth_vol = log_returns_synth.rolling(window=window).std() * np.sqrt(252)  # Anualizada
        
        # Crear series temporales para la volatilidad
        orig_vol_series = pd.Series(orig_vol.values, index=df_adjacent.index[window:])
        synth_vol_series = pd.Series(synth_vol.values, index=synthetic_df.index[window:])
        
        # Graficar volatilidad
        plt.plot(orig_vol_series.index, orig_vol_series, color='blue', 
                label=f'Volatilidad original (ventana={window} días)', linewidth=2)
        plt.plot(synth_vol_series.index, synth_vol_series, color='green', 
                label=f'Volatilidad sintética (ventana={window} días)', linewidth=2)
        
        # Resaltar período plano
        plt.axvspan(flat_period_info['start_date'], flat_period_info['end_date'], 
                   color='lightyellow', alpha=0.3, label='Período plano')
        
        plt.title('Comparación de Volatilidad Anualizada', fontsize=14)
        plt.xlabel('Fecha', fontsize=12)
        plt.ylabel('Volatilidad (anualizada)', fontsize=12)
        plt.legend()
        
        # Formatear eje x con fechas
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('bitcoin_volatility_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 5. Visualización de datos de volumen si están disponibles
        vol_col = None
        if 'Vol.' in synthetic_df.columns:
            vol_col = 'Vol.'
        elif 'Volume' in synthetic_df.columns:
            vol_col = 'Volume'
        
        if vol_col is not None:
            plt.figure(figsize=(15, 6))
            
            # Datos originales antes y después
            if vol_col in df_before.columns:
                plt.plot(df_before.index, df_before[vol_col], color='blue', 
                        label='Volumen original (antes)', linewidth=2)
            
            if vol_col in df_flat.columns:
                plt.plot(df_flat.index, df_flat[vol_col], color='red', 
                        label='Volumen original (período plano)', linewidth=2, alpha=0.5)
            
            # Datos sintéticos
            plt.plot(synthetic_df.index, synthetic_df[vol_col], color='green', 
                    label='Volumen sintético', linewidth=2.5)
            
            if vol_col in df_after.columns:
                plt.plot(df_after.index, df_after[vol_col], color='blue', 
                        label='Volumen original (después)', linewidth=2)
            
            # Resaltar período plano
            plt.axvspan(flat_period_info['start_date'], flat_period_info['end_date'], 
                       color='lightyellow', alpha=0.3, label='Período plano')
            
            plt.title('Comparación de Volumen', fontsize=14)
            plt.xlabel('Fecha', fontsize=12)
            plt.ylabel('Volumen', fontsize=12)
            plt.legend()
            
            # Formatear eje x con fechas
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('bitcoin_volume_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # 6. Tabla de estadísticas de validación
        print("\n========== RESUMEN DE VALIDACIÓN ==========")
        print(f"RESULTADO: {'PASÓ' if validation_results['passed_validation'] else 'NO PASÓ'} LA VALIDACIÓN")
        
        if not validation_results['passed_validation'] and 'failure_reason' in validation_results:
            print(f"Motivo: {validation_results['failure_reason']}")
        
        # Mostrar estadísticas clave
        print("\nEstadísticas del precio:")
        synth_stats = validation_results['synthetic_stats']
        
        print(f"{'Métrica':<20} {'Original':<15} {'Sintético':<15} {'Diff %':<10}")
        print("-" * 60)
        
        for key in ['mean', 'median', 'std', 'min', 'max']:
            if key in synth_stats:
                orig_val = data_dict['df_before'][price_col].append(data_dict['df_after'][price_col]).mean() if key == 'mean' else None
                orig_val = data_dict['df_before'][price_col].append(data_dict['df_after'][price_col]).median() if key == 'median' else orig_val
                orig_val = data_dict['df_before'][price_col].append(data_dict['df_after'][price_col]).std() if key == 'std' else orig_val
                orig_val = data_dict['df_before'][price_col].append(data_dict['df_after'][price_col]).min() if key == 'min' else orig_val
                orig_val = data_dict['df_before'][price_col].append(data_dict['df_after'][price_col]).max() if key == 'max' else orig_val
                
                synth_val = synth_stats[key]
                
                if orig_val is not None:
                    pct_diff = ((synth_val - orig_val) / orig_val * 100) if orig_val != 0 else float('inf')
                    print(f"{key.capitalize():<20} {orig_val:<15.4f} {synth_val:<15.4f} {pct_diff:<10.2f}%")
        
        if 'daily_volatility' in synth_stats:
            # Calcular volatilidad diaria en datos originales
            df_adjacent = pd.concat([df_before, df_after])
            log_returns_orig = np.log(df_adjacent[price_col] / df_adjacent[price_col].shift(1)).dropna()
            orig_vol = log_returns_orig.std()
            
            synth_vol = synth_stats['daily_volatility']
            pct_diff = ((synth_vol - orig_vol) / orig_vol * 100) if orig_vol != 0 else float('inf')
            
            print(f"{'Volatilidad diaria':<20} {orig_vol:<15.4f} {synth_vol:<15.4f} {pct_diff:<10.2f}%")
        
        # 7. Resaltar información de continuidad en los bordes
        if 'edge_continuity' in validation_results:
            print("\nContinuidad en los bordes:")
            for key, value in validation_results['edge_continuity'].items():
                print(f"{key}: {value:.2f}%")
        
    except Exception as e:
        print(f"Error en la visualización: {str(e)}")


def create_corrected_dataset(df, synthetic_df, start_date='2010-07-18', end_date='2010-10-16'):
    """
    Crea un dataset corregido reemplazando el período plano con datos sintéticos.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame original con todos los datos
    synthetic_df : pandas.DataFrame
        DataFrame con los datos sintéticos para el período plano
    start_date : str
        Fecha de inicio del período plano en formato 'YYYY-MM-DD'
    end_date : str
        Fecha de fin del período plano en formato 'YYYY-MM-DD'
    
    Retorna:
    --------
    pandas.DataFrame
        DataFrame corregido con datos sintéticos en el período plano
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
        
        # Crear una copia del DataFrame original
        corrected_df = df.copy()
        
        # Crear máscara para el período plano
        flat_period_mask = (corrected_df.index >= start_date) & (corrected_df.index <= end_date)
        
        # Determinar columna de precio
        if 'Price' in df.columns:
            price_col = 'Price'
        elif 'Close' in df.columns:
            price_col = 'Close'
        else:
            price_col = df.select_dtypes(include=[np.number]).columns[0]
            print(f"Advertencia: Usando '{price_col}' como columna de precio.")
        
        # Reemplazar datos para cada fecha en el período plano
        for date in synthetic_df.index:
            if date in corrected_df.index:
                # Reemplazar precio
                corrected_df.loc[date, price_col] = synthetic_df.loc[date, price_col]
                
                # Reemplazar también otras columnas si están disponibles
                for col in synthetic_df.columns:
                    if col != price_col and col in corrected_df.columns:
                        corrected_df.loc[date, col] = synthetic_df.loc[date, col]
        
        # Verificar continuidad en los bordes
        if start_date in corrected_df.index and (start_date - pd.Timedelta(days=1)) in corrected_df.index:
            before_pct_change = (corrected_df.loc[start_date, price_col] - 
                               corrected_df.loc[start_date - pd.Timedelta(days=1), price_col]) / \
                               corrected_df.loc[start_date - pd.Timedelta(days=1), price_col] * 100
            print(f"Cambio porcentual en el borde inicial: {before_pct_change:.2f}%")
        
        if end_date in corrected_df.index and (end_date + pd.Timedelta(days=1)) in corrected_df.index:
            after_pct_change = (corrected_df.loc[end_date + pd.Timedelta(days=1), price_col] - 
                              corrected_df.loc[end_date, price_col]) / \
                              corrected_df.loc[end_date, price_col] * 100
            print(f"Cambio porcentual en el borde final: {after_pct_change:.2f}%")
        
        # Contar registros corregidos
        replaced_count = flat_period_mask.sum()
        print(f"Se reemplazaron {replaced_count} registros con datos sintéticos")
        
        return corrected_df
    
    except Exception as e:
        print(f"Error al crear dataset corregido: {str(e)}")
        return df  # Devolver el DataFrame original en caso de error


def visualize_complete_correction(original_df, corrected_df, start_date='2010-07-18', end_date='2010-10-16',
                               window_size=60):
    """
    Visualiza el dataset completo antes y después de la corrección.
    
    Parámetros:
    -----------
    original_df : pandas.DataFrame
        DataFrame original con datos sin corrección
    corrected_df : pandas.DataFrame
        DataFrame con datos corregidos
    start_date : str
        Fecha de inicio del período plano en formato 'YYYY-MM-DD'
    end_date : str
        Fecha de fin del período plano en formato 'YYYY-MM-DD'
    window_size : int
        Número de días a mostrar antes y después del período plano
    """
    try:
        # Convertir fechas a datetime si son strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        
        # Crear ventana para visualización
        window_start = start_date - pd.Timedelta(days=window_size)
        window_end = end_date + pd.Timedelta(days=window_size)
        
        # Filtrar datos para la ventana de visualización
        mask = (original_df.index >= window_start) & (original_df.index <= window_end)
        orig_window = original_df[mask]
        corr_window = corrected_df[mask]
        
        # Determinar columna de precio
        if 'Price' in original_df.columns:
            price_col = 'Price'
        elif 'Close' in original_df.columns:
            price_col = 'Close'
        else:
            price_col = original_df.select_dtypes(include=[np.number]).columns[0]
        
        # Visualizar serie de precios
        plt.figure(figsize=(15, 8))
        
        plt.plot(orig_window.index, orig_window[price_col], color='red', 
                label='Original (con período plano)', linewidth=2, alpha=0.7)
        plt.plot(corr_window.index, corr_window[price_col], color='green', 
                label='Corregido', linewidth=2.5)
        
        # Resaltar período plano
        plt.axvspan(start_date, end_date, color='lightyellow', alpha=0.3, 
                   label='Período corregido')
        
        plt.title('Comparación de Datos Originales vs. Corregidos', fontsize=16)
        plt.xlabel('Fecha', fontsize=14)
        plt.ylabel(f'{price_col} (USD)', fontsize=14)
        plt.legend()
        
        # Formatear eje x con fechas
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
        plt.xticks(rotation=45)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('bitcoin_complete_correction.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Visualizar cambio porcentual si está disponible
        if 'Change %' in original_df.columns and 'Change %' in corrected_df.columns:
            plt.figure(figsize=(15, 6))
            
            plt.plot(orig_window.index, orig_window['Change %'], color='red', 
                    label='Cambio % Original', linewidth=2, alpha=0.7)
            plt.plot(corr_window.index, corr_window['Change %'], color='green', 
                    label='Cambio % Corregido', linewidth=2.5)
            
            # Resaltar período plano
            plt.axvspan(start_date, end_date, color='lightyellow', alpha=0.3, 
                       label='Período corregido')
            
            plt.title('Comparación de Cambio Porcentual: Original vs. Corregido', fontsize=14)
            plt.xlabel('Fecha', fontsize=12)
            plt.ylabel('Cambio %', fontsize=12)
            plt.legend()
            
            # Formatear eje x con fechas
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('bitcoin_change_pct_correction.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Calcular mejora en volatilidad
        original_std = orig_window[price_col].std()
        corrected_std = corr_window[price_col].std()
        std_improvement = ((corrected_std / original_std) - 1) * 100
        
        print("\n========== MEJORA EN LA CALIDAD DE DATOS ==========")
        print(f"Desviación estándar original: {original_std:.4f}")
        print(f"Desviación estándar corregida: {corrected_std:.4f}")
        print(f"Mejora en variabilidad: {std_improvement:.2f}%")
        
        # Calcular retornos logarítmicos y comparar distribuciones
        log_returns_orig = np.log(orig_window[price_col] / orig_window[price_col].shift(1)).dropna()
        log_returns_corr = np.log(corr_window[price_col] / corr_window[price_col].shift(1)).dropna()
        
        # Test de normalidad
        shapiro_orig = stats.shapiro(log_returns_orig)
        shapiro_corr = stats.shapiro(log_returns_corr)
        
        print(f"\nTest de normalidad (Shapiro-Wilk) para retornos logarítmicos:")
        print(f"Original: p-value = {shapiro_orig[1]:.6f} ({'Normal' if shapiro_orig[1] > 0.05 else 'No Normal'})")
        print(f"Corregido: p-value = {shapiro_corr[1]:.6f} ({'Normal' if shapiro_corr[1] > 0.05 else 'No Normal'})")
        
    except Exception as e:
        print(f"Error en la visualización completa: {str(e)}")


# Si se ejecuta como script independiente
if __name__ == "__main__":
    print("Módulo de visualización para datos sintéticos de Bitcoin")
    print("Importa las funciones en tu notebook para usarlas.")
    print("Ejemplo de uso:")
    print("from bitcoin_visualization import visualize_results, create_corrected_dataset, visualize_complete_correction")