"""
Bitcoin Period Filter (2011-2017)
Script para filtrar datos de Bitcoin y mantener solo el período 2011-2017
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def filter_bitcoin_period(df, start_date='2011-01-01', end_date='2017-12-31', date_column='Date'):
    """
    Filtra un DataFrame de Bitcoin para incluir solo datos entre las fechas especificadas.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame con los datos de Bitcoin
    start_date : str
        Fecha de inicio en formato 'YYYY-MM-DD' (por defecto: '2011-01-01')
    end_date : str
        Fecha de fin en formato 'YYYY-MM-DD' (por defecto: '2017-12-31')
    date_column : str
        Nombre de la columna de fecha (por defecto: 'Date')
        
    Retorna:
    --------
    pandas.DataFrame
        DataFrame filtrado con datos solo del período especificado
    """
    # Hacer una copia para no modificar el original
    filtered_df = df.copy()
    
    # Verificar si la columna de fecha está presente
    if date_column not in filtered_df.columns and filtered_df.index.name != date_column:
        print(f"Advertencia: Columna de fecha '{date_column}' no encontrada.")
        if isinstance(filtered_df.index, pd.DatetimeIndex):
            print(f"Usando el índice como fecha.")
            has_date_index = True
        else:
            return filtered_df
    else:
        has_date_index = (filtered_df.index.name == date_column)
    
    # Convertir fecha a datetime si es necesario
    if not has_date_index:
        filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors='coerce')
        
        # Eliminar filas con fechas inválidas
        invalid_dates = filtered_df[date_column].isna()
        if invalid_dates.any():
            n_invalid = invalid_dates.sum()
            print(f"Eliminando {n_invalid} filas con fechas inválidas.")
            filtered_df = filtered_df[~invalid_dates]
    
    # Convertir fechas de inicio y fin a datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Aplicar filtro
    if has_date_index:
        filtered_df = filtered_df.loc[start:end].copy()
    else:
        mask = (filtered_df[date_column] >= start) & (filtered_df[date_column] <= end)
        filtered_df = filtered_df.loc[mask].copy()
    
    # Estadísticas sobre el filtrado
    print(f"Período filtrado: {start.strftime('%Y-%m-%d')} a {end.strftime('%Y-%m-%d')}")
    print(f"Registros en el período: {len(filtered_df)}")
    
    return filtered_df

def load_and_filter_bitcoin_csv(file_path, start_date='2011-01-01', end_date='2017-12-31', date_column='Date', set_index=True):
    """
    Carga un archivo CSV con datos de Bitcoin y filtra para incluir solo el período especificado.
    
    Parámetros:
    -----------
    file_path : str
        Ruta al archivo CSV con los datos de Bitcoin
    start_date : str
        Fecha de inicio en formato 'YYYY-MM-DD' (por defecto: '2011-01-01')
    end_date : str
        Fecha de fin en formato 'YYYY-MM-DD' (por defecto: '2017-12-31')
    date_column : str
        Nombre de la columna de fecha (por defecto: 'Date')
    set_index : bool
        Si se debe establecer la columna de fecha como índice (por defecto: True)
        
    Retorna:
    --------
    pandas.DataFrame
        DataFrame filtrado con datos solo del período especificado
    """
    try:
        # Cargar datos
        print(f"Cargando datos desde {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Datos cargados: {len(df)} registros")
        
        # Establecer fecha como índice si se solicita
        if set_index and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)
        
        # Filtrar por período
        filtered_df = filter_bitcoin_period(df, start_date, end_date, date_column)
        
        return filtered_df
        
    except Exception as e:
        print(f"Error al cargar o filtrar datos: {str(e)}")
        return pd.DataFrame()

def save_filtered_data(df, output_path, index=True):
    """
    Guarda el DataFrame filtrado en un archivo CSV.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame a guardar
    output_path : str
        Ruta donde guardar el archivo CSV
    index : bool
        Si se debe incluir el índice en el CSV (por defecto: True)
    """
    try:
        df.to_csv(output_path, index=index)
        print(f"Datos guardados en {output_path}")
        return True
    except Exception as e:
        print(f"Error al guardar datos: {str(e)}")
        return False

def plot_filtered_data(df, price_column='Price', title='Bitcoin Price (2011-2017)'):
    """
    Visualiza los datos filtrados.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame filtrado con los datos de Bitcoin
    price_column : str
        Nombre de la columna de precio (por defecto: 'Price')
    title : str
        Título del gráfico (por defecto: 'Bitcoin Price (2011-2017)')
    """
    # Verificar que tenemos un índice de fecha
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Advertencia: El índice no es de tipo fecha. Intentando convertir...")
        df.index = pd.to_datetime(df.index)
    
    # Verificar que existe la columna de precio
    if price_column not in df.columns:
        # Buscar alternativas
        alternatives = ['Close', 'close', 'Adj Close', 'adj_close']
        for alt in alternatives:
            if alt in df.columns:
                price_column = alt
                print(f"Usando columna '{price_column}' para el precio.")
                break
        else:
            print(f"Error: No se encontró columna de precio. Columnas disponibles: {df.columns.tolist()}")
            return
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[price_column])
    plt.title(title)
    plt.xlabel('Fecha')
    plt.ylabel(f'Precio (USD)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Ejemplo de uso si se ejecuta como script
if __name__ == "__main__":
    print("Módulo de filtrado de datos de Bitcoin (2011-2017)")
    print("Este script está diseñado para ser importado desde un notebook.")
    print("\nEjemplo de uso:")
    print("from src.bitcoin_period_filter import load_and_filter_bitcoin_csv, save_filtered_data, plot_filtered_data")
    print("df = load_and_filter_bitcoin_csv('ruta/a/Bitcoin_History.csv')")
    print("plot_filtered_data(df)")
    print("save_filtered_data(df, 'ruta/a/Bitcoin_2011_2017.csv')")