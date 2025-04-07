import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

class BitcoinDataCleaner:
    """
    Clase para limpiar e imputar datos de Bitcoin.
    Maneja la conversión de formatos y la imputación de valores nulos.
    """
    
    def __init__(self, verbose=True):
        """
        Inicializa el limpiador de datos.
        
        Args:
            verbose (bool): Si se imprime información detallada del proceso
        """
        self.verbose = verbose
    
    def load_data(self, file_path):
        """
        Carga los datos desde un archivo CSV.
        
        Args:
            file_path (str): Ruta al archivo CSV
            
        Returns:
            pd.DataFrame: DataFrame con los datos cargados
        """
        try:
            df = pd.read_csv(file_path)
            
            if self.verbose:
                print(f"Datos cargados desde {file_path}")
                print(f"Dimensiones: {df.shape}")
                print("Primeras filas:")
                print(df.head())
                print("\nTipos de datos:")
                print(df.dtypes)
            
            return df
        except Exception as e:
            print(f"Error al cargar los datos: {str(e)}")
            return pd.DataFrame()
    
    def convert_date_column(self, df, date_col='Date'):
        """
        Convierte la columna de fecha al formato adecuado y la establece como índice.
        
        Args:
            df (pd.DataFrame): DataFrame a procesar
            date_col (str): Nombre de la columna de fecha
            
        Returns:
            pd.DataFrame: DataFrame con la fecha como índice
        """
        if date_col not in df.columns:
            if self.verbose:
                print(f"Advertencia: No se encontró columna '{date_col}'")
            return df
        
        # Convertir a datetime y establecer como índice
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        
        # Ordenar por fecha (más reciente primero, como aparece en tu ejemplo)
        df = df.sort_index(ascending=False)
        
        if self.verbose:
            print(f"Columna '{date_col}' convertida a datetime y establecida como índice")
            print("DataFrame ordenado por fecha (descendente)")
        
        return df
    
    def convert_numeric_columns(self, df):
        """
        Convierte columnas numéricas a su formato correcto.
        
        Args:
            df (pd.DataFrame): DataFrame a procesar
            
        Returns:
            pd.DataFrame: DataFrame con columnas convertidas
        """
        df_clean = df.copy()
        
        # Convertir columnas básicas de precios
        price_cols = ['Price', 'Open', 'High', 'Low']
        for col in price_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                if self.verbose:
                    print(f"Columna '{col}' convertida a numérica")
        
        # Convertir volumen (manejar sufijos K, M, B)
        if 'Vol.' in df_clean.columns:
            # Crear una copia de la columna original para referencia
            vol_original = df_clean['Vol.'].copy()
            
            # Función para convertir volumen con sufijos K, M, B
            def parse_volume(vol_str):
                if pd.isna(vol_str) or not isinstance(vol_str, str):
                    return vol_str
                
                vol_str = vol_str.strip()
                
                # Manejar caso de guión solo ('-')
                if vol_str == '-':
                    return 0.0  # O np.nan si prefieres marcarlo como faltante
                
                # Manejar sufijos K (miles), M (millones), B (billones)
                if vol_str.endswith('K'):
                    return float(vol_str.replace('K', '')) * 1_000
                elif vol_str.endswith('M'):
                    return float(vol_str.replace('M', '')) * 1_000_000
                elif vol_str.endswith('B'):
                    return float(vol_str.replace('B', '')) * 1_000_000_000
                else:
                    return float(vol_str.replace(',', ''))
            
            # Aplicar la función de conversión
            df_clean['Vol.'] = df_clean['Vol.'].apply(parse_volume)
            
            if self.verbose:
                print("Columna 'Vol.' convertida a numérica (con manejo de sufijos K, M, B)")
        
        # Convertir cambio porcentual
        if 'Change %' in df_clean.columns:
            # Eliminar el símbolo % y convertir a decimal
            df_clean['Change %'] = df_clean['Change %'].str.rstrip('%').astype('float') / 100.0
            
            if self.verbose:
                print("Columna 'Change %' convertida a decimal")
        
        return df_clean
    
    def check_missing_values(self, df):
        """
        Verifica los valores nulos en el DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame a verificar
        """
        missing = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df)) * 100
        
        missing_info = pd.DataFrame({
            'Valores Nulos': missing,
            'Porcentaje (%)': missing_pct
        })
        
        print("\nInformación de valores faltantes:")
        print(missing_info[missing_info['Valores Nulos'] > 0])
        
        # Visualizar valores nulos
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
        plt.title('Mapa de Valores Nulos')
        plt.tight_layout()
        plt.show()
    
    def impute_missing_values(self, df, method='combined'):
        """
        Imputa valores nulos usando varios métodos.
        
        Args:
            df (pd.DataFrame): DataFrame a procesar
            method (str): Método de imputación ('combined', 'interpolation', 'forward_fill')
            
        Returns:
            pd.DataFrame: DataFrame con valores imputados
        """
        df_imputed = df.copy()
        
        if method == 'combined':
            # Método combinado (recomendado):
            # 1. Interpolación lineal para huecos cortos (hasta 5 días)
            # 2. Forward fill para huecos restantes
            # 3. Backward fill para valores al inicio
            
            if self.verbose:
                print("\nAplicando método de imputación combinado...")
            
            # Paso 1: Interpolación lineal para huecos cortos
            df_imputed = df_imputed.interpolate(method='linear', limit=5, limit_direction='both')
            
            # Verificar valores aún faltantes
            still_missing = df_imputed.isnull().sum()
            if still_missing.sum() > 0 and self.verbose:
                print("Después de interpolación, quedan valores nulos:")
                print(still_missing[still_missing > 0])
            
            # Paso 2: Forward fill para huecos restantes
            df_imputed = df_imputed.fillna(method='ffill')
            
            # Paso 3: Backward fill para cualquier valor al inicio
            df_imputed = df_imputed.fillna(method='bfill')
        
        elif method == 'interpolation':
            # Solo interpolación
            df_imputed = df_imputed.interpolate(method='linear')
        
        elif method == 'forward_fill':
            # Solo forward fill
            df_imputed = df_imputed.fillna(method='ffill')
            # Backward fill para valores al inicio
            df_imputed = df_imputed.fillna(method='bfill')
        
        if self.verbose:
            still_missing = df_imputed.isnull().sum().sum()
            print(f"Valores nulos después de imputación: {still_missing}")
        
        return df_imputed
    
    def visualize_imputation_results(self, df_original, df_imputed, columns=None):
        """
        Visualiza los resultados de la imputación comparando los datos originales y los imputados.
        
        Args:
            df_original (pd.DataFrame): DataFrame original
            df_imputed (pd.DataFrame): DataFrame con valores imputados
            columns (list): Lista de columnas a visualizar (None para todas)
        """
        if columns is None:
            # Seleccionar columnas numéricas excepto las categóricas
            columns = df_original.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        n_cols = len(columns)
        if n_cols == 0:
            print("No hay columnas para visualizar.")
            return
        
        fig, axes = plt.subplots(n_cols, 1, figsize=(12, 5 * n_cols))
        
        # Ajustar para el caso de una sola columna
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(columns):
            # Graficar datos originales
            axes[i].plot(df_original.index, df_original[col], 'o', color='red', 
                        label='Originales', alpha=0.5, markersize=4)
            
            # Graficar datos imputados
            axes[i].plot(df_imputed.index, df_imputed[col], '-', color='blue', 
                        label='Imputados')
            
            # Resaltar valores imputados
            mask_null = df_original[col].isnull()
            if mask_null.sum() > 0:
                axes[i].plot(df_imputed.index[mask_null], df_imputed.loc[mask_null, col], 
                            'o', color='green', label='Valores Imputados', markersize=6)
            
            axes[i].set_title(f'Imputación para {col}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Formato para el eje X (fechas)
            plt.setp(axes[i].xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.show()
    
    def save_cleaned_data(self, df, output_file):
        """
        Guarda el DataFrame limpio en un archivo CSV.
        
        Args:
            df (pd.DataFrame): DataFrame a guardar
            output_file (str): Ruta del archivo de salida
        """
        try:
            df.to_csv(output_file)
            if self.verbose:
                print(f"\nDatos limpios guardados en {output_file}")
                print(f"Dimensiones finales: {df.shape}")
        except Exception as e:
            print(f"Error al guardar los datos: {str(e)}")
    
    def clean_data_pipeline(self, input_file, output_file):
        """
        Pipeline completo para limpiar los datos.
        
        Args:
            input_file (str): Ruta al archivo de entrada
            output_file (str): Ruta al archivo de salida
            
        Returns:
            pd.DataFrame: DataFrame limpio
        """
        print("\n=== INICIANDO PROCESO DE LIMPIEZA E IMPUTACIÓN ===\n")
        
        # 1. Cargar datos
        df = self.load_data(input_file)
        if df.empty:
            return df
        
        # 2. Convertir columna de fecha
        df = self.convert_date_column(df)
        
        # 3. Convertir columnas numéricas
        df_clean = self.convert_numeric_columns(df)
        
        # 4. Verificar valores nulos
        if self.verbose:
            self.check_missing_values(df_clean)
        
        # 5. Imputar valores nulos
        df_imputed = self.impute_missing_values(df_clean, method='combined')
        
        # 6. Visualizar resultados de imputación
        if self.verbose:
            self.visualize_imputation_results(df_clean, df_imputed)
        
        # 7. Guardar datos limpios
        self.save_cleaned_data(df_imputed, output_file)
        
        print("\n=== PROCESO DE LIMPIEZA E IMPUTACIÓN COMPLETADO ===\n")
        
        return df_imputed

# Ejemplo de uso
if __name__ == "__main__":
    # Definir rutas
    input_file = "../BITCOIN/Bitcoin_History.csv"  # Ajustar según tu estructura
    output_file = "../BITCOIN/Bitcoin_History_Clean.csv"
    
    # Crear instancia y ejecutar pipeline
    cleaner = BitcoinDataCleaner(verbose=True)
    clean_df = cleaner.clean_data_pipeline(input_file, output_file)