import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class BitcoinPreprocessor:
    """
    Pipeline de preprocesamiento para datos de Bitcoin enfocado en preparación para modelado.
    Versión mejorada con manejo de errores y validaciones adicionales.
    """
    
    def __init__(self, remove_high_corr=True, corr_threshold=0.8, scale_features=True,
                 ensure_stationarity=True, target_col='return_daily'):
        """
        Inicializar el preprocesador con parámetros configurables.
        
        Args:
            remove_high_corr (bool): Si se deben eliminar features altamente correlacionadas
            corr_threshold (float): Umbral para considerar alta correlación
            scale_features (bool): Si se deben escalar las features
            ensure_stationarity (bool): Si se debe verificar y transformar para asegurar estacionariedad
            target_col (str): Nombre de la columna objetivo
        """
        self.remove_high_corr = remove_high_corr
        self.corr_threshold = corr_threshold
        self.scale_features = scale_features
        self.ensure_stationarity = ensure_stationarity
        self.target_col = target_col
        self.high_corr_pairs = []
        self.scaler = StandardScaler()
        self.non_stationary_cols = []
        self.selected_features = []
        
    def fit_transform(self, df):
        """
        Ajusta el preprocesador a los datos y los transforma.
        
        Args:
            df (pd.DataFrame): DataFrame con los datos de Bitcoin
            
        Returns:
            pd.DataFrame: DataFrame procesado listo para modelado
        """
        # Verificar que el DataFrame no esté vacío
        if df.empty:
            print("Error: El DataFrame está vacío.")
            return df
            
        # Hacemos una copia para no modificar el original
        processed_df = df.copy()
        
        # Imprimir información básica
        print(f"DataFrame original: {processed_df.shape} filas y columnas")
        print(f"Columnas disponibles: {processed_df.columns.tolist()}")
        
        # 1. Verificar valores nulos
        self._handle_missing_values(processed_df)
        
        # 2. Analizar y tratar correlaciones
        if self.remove_high_corr:
            processed_df = self._handle_correlations(processed_df)
        
        # 3. Verificar y transformar para estacionariedad
        if self.ensure_stationarity:
            processed_df = self._ensure_stationarity(processed_df)
        
        # 4. Escalar features (si es necesario)
        if self.scale_features:
            processed_df = self._scale_features(processed_df)
        
        print(f"DataFrame procesado: {processed_df.shape} filas y columnas")
        return processed_df
    
    def _handle_missing_values(self, df):
        """
        Maneja los valores nulos en el dataset.
        """
        # Verificar valores nulos
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        
        if not cols_with_nulls.empty:
            print("\nValores nulos por columna:")
            print(cols_with_nulls)
            
            # Estrategias para manejar nulos:
            # 1. Para valores de precios: forward fill (usar último valor disponible)
            price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
            existing_price_cols = [col for col in price_cols if col in df.columns]
            if existing_price_cols:
                df[existing_price_cols] = df[existing_price_cols].fillna(method='ffill')
                print(f"Se aplicó forward fill a las columnas de precios: {existing_price_cols}")
            
            # 2. Para indicadores técnicos: puede ser apropiado llenar con mediana o media
            technical_cols = [col for col in df.columns if col not in price_cols + ['Volume', 'Date']]
            for col in technical_cols:
                if col in df.columns and df[col].isnull().sum() > 0:
                    # Usar mediana para evitar efectos de outliers
                    df[col] = df[col].fillna(df[col].median())
                    print(f"Se llenaron nulos en '{col}' con la mediana")
            
            # Para Volume, llenar con 0 o la mediana (dependiendo del contexto)
            if 'Volume' in df.columns and df['Volume'].isnull().sum() > 0:
                df['Volume'] = df['Volume'].fillna(df['Volume'].median())
                print("Se llenaron nulos en 'Volume' con la mediana")
        else:
            print("No se encontraron valores nulos en el dataset.")
            
        return df
    
    def _handle_correlations(self, df):
        """
        Identifica y maneja features altamente correlacionadas.
        """
        # Seleccionar solo columnas numéricas
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_cols) < 2:
            print("Advertencia: No hay suficientes columnas numéricas para analizar correlaciones.")
            return df
            
        # Calcular matriz de correlación
        corr_matrix = df[numeric_cols].corr()
        
        # Identificar pares altamente correlacionados
        self.high_corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                if abs(corr_matrix.iloc[i, j]) > self.corr_threshold:
                    self.high_corr_pairs.append((numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i, j]))
        
        # Reportar pares correlacionados
        if self.high_corr_pairs:
            print(f"\nPares de features con alta correlación (>{self.corr_threshold}):")
            for pair in self.high_corr_pairs:
                print(f"{pair[0]} y {pair[1]}: {pair[2]:.4f}")
        
            # Estrategia para eliminar: por cada par correlacionado, mantener una y eliminar otra
            to_remove = set()
            for pair in self.high_corr_pairs:
                # Si una de las columnas es nuestro target, no la eliminamos
                if pair[0] == self.target_col:
                    to_remove.add(pair[1])
                elif pair[1] == self.target_col:
                    to_remove.add(pair[0])
                # Si 'Close' está en el par, preservar 'Close' por ser más intuitivo
                elif pair[0] == 'Close':
                    to_remove.add(pair[1])
                elif pair[1] == 'Close':
                    to_remove.add(pair[0])
                # De lo contrario, eliminar la segunda columna del par
                else:
                    to_remove.add(pair[1])
            
            print(f"\nColumnas eliminadas por alta correlación: {to_remove}")
            df.drop(columns=list(to_remove), inplace=True)
        else:
            print("\nNo se encontraron pares de features con correlación alta.")
        
        return df
    
    def _ensure_stationarity(self, df):
        """
        Verifica y transforma columnas para asegurar estacionariedad.
        """
        # Columnas a verificar (precios y retornos principalmente)
        cols_to_check = [col for col in df.columns if col not in ['Date']]
        cols_to_check = [col for col in cols_to_check if df[col].dtype in ['float64', 'int64']]
        
        if not cols_to_check:
            print("Advertencia: No hay columnas numéricas para verificar estacionariedad.")
            return df
        
        self.non_stationary_cols = []
        transformed_df = df.copy()
        
        for col in cols_to_check:
            series = df[col].dropna()
            if len(series) < 20:
                print(f"La columna '{col}' tiene menos de 20 valores no nulos, se omite el test de estacionariedad.")
                continue
                
            # Test de Dickey-Fuller
            try:
                result = adfuller(series)
                is_stationary = result[1] < 0.05
                
                if not is_stationary:
                    self.non_stationary_cols.append(col)
                    print(f"Serie no estacionaria: {col} (p-value: {result[1]:.4f})")
                    
                    # Si es una serie de precios, diferenciación
                    if col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                        transformed_df[f'{col}_diff'] = df[col].diff()
                        # Mantener la columna original pero marcarla
                        print(f"  → Creada nueva feature: {col}_diff (diferenciada)")
                    
                    # Para otras columnas, posiblemente log-transformación
                    elif df[col].min() > 0:  # Solo aplicar log a valores positivos
                        transformed_df[f'{col}_log'] = np.log(df[col])
                        print(f"  → Creada nueva feature: {col}_log (log-transformada)")
            except Exception as e:
                print(f"Error al verificar estacionariedad para '{col}': {str(e)}")
        
        return transformed_df
    
    def _scale_features(self, df):
        """
        Escala las features numéricas usando StandardScaler.
        """
        # Separar target y features
        if self.target_col in df.columns:
            target = df[self.target_col]
            features_df = df.drop(columns=[self.target_col])
        else:
            print(f"Advertencia: Columna target '{self.target_col}' no encontrada en el dataset.")
            target = None
            features_df = df.copy()
        
        # Seleccionar solo columnas numéricas para escalar
        numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
        
        if len(numeric_cols) == 0:
            print("Advertencia: No hay columnas numéricas para escalar.")
            if target is not None:
                features_df[self.target_col] = target
            return features_df
        
        # Eliminar columnas con todos valores nulos
        cols_to_scale = []
        for col in numeric_cols:
            if not features_df[col].isnull().all():
                cols_to_scale.append(col)
        
        if not cols_to_scale:
            print("Advertencia: Todas las columnas numéricas tienen solo valores nulos.")
            if target is not None:
                features_df[self.target_col] = target
            return features_df
            
        # Aplicar escalado
        try:
            scaled_data = self.scaler.fit_transform(features_df[cols_to_scale])
            scaled_df = pd.DataFrame(scaled_data, columns=cols_to_scale, index=features_df.index)
            
            # Reemplazar columnas originales con las escaladas
            for col in cols_to_scale:
                features_df[col] = scaled_df[col]
                
            print(f"Se escalaron {len(cols_to_scale)} columnas numéricas.")
        except Exception as e:
            print(f"Error al escalar features: {str(e)}")
        
        # Reincorporar target si existía
        if target is not None:
            features_df[self.target_col] = target
        
        return features_df
    
    def load_data(self, file_path, decimal_separator='.', thousands_separator=None):
        """
        Carga los datos de un archivo CSV con manejo de diferentes separadores decimales.
        
        Args:
            file_path (str): Ruta al archivo CSV
            decimal_separator (str): Separador decimal a usar ('.', ',')
            thousands_separator (str): Separador de miles (None, '.', ',')
            
        Returns:
            pd.DataFrame: DataFrame cargado con tipos de datos corregidos
        """
        try:
            # Si el separador decimal es una coma, usamos el parámetro decimal
            if decimal_separator == ',':
                df = pd.read_csv(file_path, decimal=',', thousands=thousands_separator)
            else:
                df = pd.read_csv(file_path, thousands=thousands_separator)
            
            # Convertir 'Date' a datetime si existe
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            
            # Verificar y corregir tipos de datos
            self._fix_data_types(df)
                
            print(f"Datos cargados exitosamente: {df.shape} filas y columnas")
            print("\nTipos de datos:")
            print(df.dtypes)
            
            return df
        except Exception as e:
            print(f"Error al cargar el archivo {file_path}: {str(e)}")
            return pd.DataFrame()  # Devolver DataFrame vacío en caso de error

    def _fix_data_types(self, df):
        """
        Corrige los tipos de datos en el DataFrame, asegurando que las columnas
        numéricas se interpreten correctamente.
        
        Args:
            df (pd.DataFrame): DataFrame a corregir
        """
        # Columnas que deberían ser numéricas
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        other_numeric_cols = ['return_daily', 'return_log', 'volatility_20d', 
                           'volume_norm', 'price_rel_max', 'price_rel_min']
        
        # Buscar todas las columnas que podrían ser numéricas
        potential_numeric_cols = price_cols + other_numeric_cols
        
        # Intentar convertir cada columna a numérica
        for col in df.columns:
            if col in potential_numeric_cols or any(keyword in col.lower() for keyword in ['price', 'return', 'volume', 'volatility']):
                try:
                    # Si la columna tiene formato de string con comas como decimales
                    if df[col].dtype == 'object':
                        # Primero intentar convertir directamente
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        print(f"Columna {col} convertida a numérica.")
                        
                        # Si hay muchos NaN después de la conversión, puede ser por separadores decimales
                        if df[col].isnull().sum() > len(df) * 0.5:  # Si más del 50% son NaN
                            # Intentar reemplazar comas por puntos y convertir de nuevo
                            if isinstance(df[col].iloc[0], str) and ',' in df[col].iloc[0]:
                                df[col] = df[col].str.replace(',', '.').astype(float)
                                print(f"Columna {col} convertida reemplazando comas por puntos.")
                except Exception as e:
                    print(f"No se pudo convertir la columna {col} a numérica: {str(e)}")
        
        # Verificar columnas que ahora son numéricas
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        print(f"\nColumnas numéricas después de la corrección: {list(numeric_cols)}")

    def save_processed_data(self, df, train_file, test_file, train_size=0.8):
        """
        Divide los datos en conjuntos de entrenamiento y prueba y los guarda.
        
        Args:
            df (pd.DataFrame): DataFrame procesado
            train_file (str): Ruta para guardar datos de entrenamiento
            test_file (str): Ruta para guardar datos de prueba
            train_size (float): Proporción para entrenamiento (0-1)
            
        Returns:
            tuple: (train_df, test_df) - DataFrames de entrenamiento y prueba
        """
        try:
            # División temporal (típica en series temporales)
            train_size = int(train_size * len(df))
            train_data = df.iloc[:train_size]
            test_data = df.iloc[train_size:]
            
            # Guardar archivos
            train_data.to_csv(train_file)
            test_data.to_csv(test_file)
            
            print(f"Datos de entrenamiento guardados en {train_file}: {train_data.shape}")
            print(f"Datos de prueba guardados en {test_file}: {test_data.shape}")
            
            return train_data, test_data
        except Exception as e:
            print(f"Error al guardar los datos procesados: {str(e)}")
            return df.iloc[:0], df.iloc[:0]  # Devolver DataFrames vacíos en caso de error

