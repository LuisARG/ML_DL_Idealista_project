# -*- coding: utf-8 -*-

import glob
import json
import os
import requests
from requests.auth import HTTPBasicAuth

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.feature_extraction import FeatureHasher

# https://developers.idealista.com/access-request
class Idealista:
    '''
    Clase que permite obtener información del portal Idealista a partir de su API.
    '''
    
    __url_token = "https://api.idealista.com/oauth/token"
    __url_search = "http://api.idealista.com/3.5/es/search?"
    __token = None
    
    def __init__(self, debug = False):
        '''
        Constructor
        Input:
            debug: si vale True muestra mensajes de debug
        '''
        self.__debug = debug
    
    def __get_has_parkingspace(self, parking_space):
        '''
        Método "privado".
        Obtiene el valor de la propiedad 'hasParkingSpace' del diccionario pasado como argumento.
        
        Argumentos:
        * parking_space: dicionario con la sigueitne estructura {'hasParkingSpace': True, 'isParkingSpaceIncludedInPrice': True}
        
        Resultado:
        * Valor de la propiedad 'hasParkingSpace'
        '''
        # 'parkingSpace': {'hasParkingSpace': True, 'isParkingSpaceIncludedInPrice': True}
        hasparkingspace = np.nan
        if pd.notnull(parking_space):
            hasparkingspace = parking_space['hasParkingSpace']
        
        return hasparkingspace
        
    def __get_is_parkingspace_inc_inprice(self, parking_space):
        '''
        Método "privado".
        Obtiene el valor de la propiedad 'isParkingSpaceIncludedInPrice' del diccionario pasado como argumento.
        
        Argumentos:
        * parking_space: dicionario con la sigueitne estructura {'hasParkingSpace': True, 'isParkingSpaceIncludedInPrice': True}
        
        Resultado:
        * Valor de la propiedad 'isParkingSpaceIncludedInPrice'
        '''
        # 'parkingSpace': {'hasParkingSpace': True, 'isParkingSpaceIncludedInPrice': True}
        inprice = np.nan
        if pd.notnull(parking_space):
            inprice = parking_space['isParkingSpaceIncludedInPrice']
        
        return inprice
        
    def __get_parkingSpacePrice(self, parking_space):
        parkingSpacePrice = np.nan
        if pd.notnull(parking_space):
            parkingSpacePrice = parking_space['parkingSpacePrice']
        
        return parkingSpacePrice
        
    def generate_token(self, api_key=None, api_secret=None):
        '''
        Genera un token necesario para usar las funciones ofrecidas por la API de Idealista.
        Este token es necesario para invocar al método 'search'.
        
        Argumentos:
        * api_key: api_key
        * api_secret: api_secret
        
        Resultado:
        * El token
        '''
        basic_auth = HTTPBasicAuth(api_key, api_secret)
            
        r = requests.post(self.__url_token,
                          auth=basic_auth,
                          data={"grant_type": "client_credentials"})

        token_response = json.loads(r.text)
        if self.__debug:
            print("r.text:", r.text)
        self.__token = token_response["access_token"]
        if self.__debug:
            print("Token:", self.__token)
    
    def set_token(self, token):
        '''
        Establece un token de la API de Idealista.
        Este token es necesario para invocar al método 'search'.
        
        Argumentos:
        * token: token de la API de Idealista.
        '''
        self.__token = token
        
    def search(self, center, country='es', numPage='1', maxItems='50', distance='1000', propertyType='homes', operation='sale'):
        '''
        Realiza búsquedas de propiedades en Idealista.
        
        Argumentos:
        * center: centro del radio de búsqueda en longitud/latitud
        * country: código de país, que puede tomar uno de estos valores:
            es - idealista.com
            it - idealista.it
            pt - idealista.pt
        * numPage: número de página (para paginación)
        * maxItems: número de lementos por página (para paginación)
        * distance: radio de búsqueda en metros
        * propertyType: tipo de propiedad a buscar. Posibles valores: homes, offices, premises, garages, bedrooms
        * operation: tipo de operación. Posibles valores: sale, rent
        
        Resultado:
        * JSON con el resultado de la invocación a la función search de la API de Idealista
        '''
        api_url = self.__url_search + \
            'country=' + country +\
            '&center=' + center +\
            '&numPage=' + numPage +\
            '&maxItems=' + maxItems +\
            '&distance=' + distance +\
            '&propertyType=' + propertyType +\
            '&operation=' + operation
        if self.__debug:
            print(api_url)
        
        headers = {"Authorization": "Bearer " + self.__token}
        r = requests.post(api_url, headers = headers)
        result = None
        try:
            result = json.loads(r.text)
        except Exception as ex:
            print("Error:", ex)
            print("r.text:", r.text)            
            
        return result
    
    def summary_result(self, result):
        
        summary = {
            "total": result["total"],
            "totalPages": result["totalPages"],
            "actualPage": result["actualPage"],
            "itemsPerPage": result["itemsPerPage"]
        }
        
        return summary
    
    def elementlist_tojson(self, result, file):
        '''
        Graba en un fichero el elemento "elementList" del resultado de una invocación a la función search de la API de Idealista.
        Argumentos:
        * result: resultado de una invocación a la función search de la API de Idealista
        * file: path del fichero donde graba el elemento "elementList"     
        '''
        with open(file, 'w') as outfile:
            json.dump(result["elementList"], outfile)
    
    def read_json(self, file, type='frame'):
        '''
        Genera un DataFrame a partir de un fichero con el JSON que contiene el elemento "elementList" del resultado de una invocación
        a la función search de la API de Idealista.
        Parametros:
        * file: fichero con el JSON
        * type: 'frame', 'series'
        Resultado:
        * El DataFrame
        '''
        return pd.read_json(file, orient='records', typ=type)
    
    def read_jsons(self, directory='', type='frame'):
        '''
        Lee los ficheros con el JSON que contiene el elemento "elementList" del resultado de una invocación a la función search
        de la API de Idealista y los concatena en un DataFrame de Pandas.
        
        Parametros:
        * directory: directorio donde se encuantra los ficheros
        * type: 'frame', 'series'
        Resultado:
        * El DataFrame
        '''
        list_ficheros_json = glob.glob(os.path.join(directory,'*.json'))

        list_df = [self.read_json(fich_json) for fich_json in list_ficheros_json]
        
        idealista_df = None
        if len(list_df) == 1:
            idealista_df = list_df[0].copy()
        elif len(list_df) > 1:
            idealista_df = list_df[0].copy()
            for ind in range(1,len(list_df)):
                idealista_df = idealista_df.append(list_df[ind])
            
            idealista_df.reset_index(drop=True,inplace=True)

        return idealista_df

    def clean_dataframe(self, data_frame):
        '''
        Realiza ciertas tareas para aplinar el DataFrame:
        * Extraer "subTypology" de "detailedType"
        * Extraer "hasParkingSpace" de "parkingSpace"
        * Extraer "isParkingSpaceIncludedInPrice" de "parkingSpace"      
        * Eliminar los campos aplanados "detailedType" y "parkingSpace"
        Parametros:
        * data_frame: DataFrame a limpiar.
        '''
        data_frame["subTypology"] = data_frame["detailedType"].apply(lambda x: x.get('subTypology', ''))
        # 'parkingSpace': {'hasParkingSpace': True, 'isParkingSpaceIncludedInPrice': True}
        data_frame["hasParkingSpace"] = data_frame["parkingSpace"].apply(self.__get_has_parkingspace)
        data_frame["isParkingSpaceIncludedInPrice"] = data_frame["parkingSpace"].apply(self.__get_is_parkingspace_inc_inprice)
        data_frame.drop(columns = ["detailedType","parkingSpace"], inplace = True)

class IdealistaFeatureEngineering:
    '''
    Clase con métodos para hacer tareas de FeatureEngineering sobre el dataset de Idealista.
    '''

    def __init__(self, debug = False):
        '''
        Constructor
        Input:
            debug: si vale True muestra mensajes de debug
        '''
        self.__debug = debug
    
    
    def nan_analysis(self, dataframe, display_analysis = True):
        '''
        Realiza un análisis de nans sobre el dataset de Idealista, generando un DataFrame con el resultado de dicho análisis.
        Opcionalmente hace un display del DataFrame con el resultado del análisis de nans.
        
        Parametros:
        * dataframe: DataFrame a analizar
        * display_analysis: si vale True se hace un display del DataFrame con el resultado del análisis de nans
        Resultado:
        * El DataFrame con el resultado del análisis de nans
        '''
        nro_filas = dataframe.shape[0]

        print("Valores nulos en cada columna:")
        suma_nulos_cols = dataframe.isna().sum().to_frame()
        suma_nulos_cols.columns = ['NroNulos']
        suma_nulos_cols['PorcentajeNulos'] = 100 * suma_nulos_cols['NroNulos'] / nro_filas
        suma_nulos_cols.sort_values(by='NroNulos', ascending=False, inplace=True)
        cols_con_nulos = suma_nulos_cols[suma_nulos_cols['NroNulos'] > 0]
        if display_analysis:
            display(cols_con_nulos)
        
        return cols_con_nulos
        

    def floor_str_2_number(self, floor):
        '''
        Realiza estas conversiones de un texto con abreviatura de una planta a número:
        - 'bj' (bajo) a 0
        - 'ss' (semosotano) a -1
        - 'st' (sotano) a -1
        - 'en' (entresuelo) a 0
        
        Parametros:
        * floor: valor a convertir
        Resultado:
        * valor convertido
        '''
        new_floor = floor

        map_floor = { "bj": 0, "en": 0, "ss": -1, "st": -1 }
        if not pd.isnull(floor) and isinstance(floor, str):
            valor = map_floor.get(floor)
            if valor != None:
                new_floor = valor
            else:
                new_floor = float(floor)

        return new_floor

    def fillna_floor(self, idealista_df):
        '''
        Asigna a las muestras con la variable 'floor' a nan la media de propiedades del mismo tipo:
        - Y del mismo tipo de barrio al que pertenezca la propiedad si en ese barrio hay propiedades
        - Y del mismo tipo de distrito al que pertenezca la propiedad si no hay propiedades en su barrio y sí en su distrito
        - De todo Madrid si no hay propiedades en su barrio ni en su distrito
        
        Parametros:
        * idealista_df: DataFrame con datos de Idealista
        '''
        df_floor_nan = idealista_df.loc[idealista_df['floor'].isna(),:]
        df_floor_nan.shape[0]
        for index, row in df_floor_nan.iterrows():
            #display(index)
            new_floor = np.nan
            aux = idealista_df.loc[idealista_df['floor'].notnull() &
                                   (idealista_df["codbarrio"] == row["codbarrio"]) & \
                                   (idealista_df["propertyType"] == row["propertyType"]),:]["floor"]
            #display(aux)
            if aux.shape[0] > 0:
                new_floor = aux.mean()
            else:
                aux = idealista_df.loc[idealista_df['floor'].notnull() &
                                       (idealista_df["coddistrit"] == row["coddistrit"]) & \
                                       (idealista_df["propertyType"] == row["propertyType"]),:]["floor"]
                if aux.shape[0] > 0:
                    new_floor = aux.mean()
                else:
                    aux = idealista_df.loc[idealista_df['floor'].notnull() &
                                           (idealista_df["propertyType"] == row["propertyType"]),:]["floor"]
                    new_floor = aux.mean()

            #print(new_floor)
            idealista_df.at[index,"floor"] = new_floor
    
    def fillna_haslift(self, idealista_df):
        '''
        Asigna a las muestras con la variable 'hasLift' a nan la moda de propiedades del mismo tipo:
        - Y del mismo tipo de barrio al que pertenezca la propiedad si en ese barrio hay propiedades
        - Y del mismo tipo de distrito al que pertenezca la propiedad si no hay propiedades en su barrio y sí en su distrito
        - De todo Madrid si no hay propiedades en su barrio ni en su distrito
        
        Parametros:
        * idealista_df: DataFrame con datos de Idealista
        '''
        df_haslift_nan = idealista_df.loc[idealista_df['hasLift'].isna(),:]
        df_haslift_nan.shape[0]
        for index, row in df_haslift_nan.iterrows():
            #display(index)
            new_haslift = np.nan
            aux = idealista_df.loc[idealista_df['hasLift'].notnull() &
                                   (idealista_df["codbarrio"] == row["codbarrio"]) & \
                                   (idealista_df["propertyType"] == row["propertyType"]),:]["hasLift"]
            #display(aux)
            if aux.shape[0] > 0:
                new_haslift = aux.mode()[0]
            else:
                aux = idealista_df.loc[idealista_df['hasLift'].notnull() &
                                       (idealista_df["coddistrit"] == row["coddistrit"]) & \
                                       (idealista_df["propertyType"] == row["propertyType"]),:]["hasLift"]
                if aux.shape[0] > 0:
                    new_haslift = aux.mode()[0]
                else:
                    aux = idealista_df.loc[idealista_df['hasLift'].notnull() &
                                           (idealista_df["propertyType"] == row["propertyType"]),:]["hasLift"]
                    new_haslift = aux.mode()[0]

            #print(new_haslift)
            idealista_df.at[index,"hasLift"] = new_haslift
        
    def feateures_bool_2_number(self, data_frame):
        '''
        Convierte las columnas booleanas de un DataFrame a ceors y unos.
        
        Parametros:
        * data_frame: DataFrame con datos a convertir
        '''
        for index, value in data_frame.dtypes.items():
            if str(value) == 'bool':
                data_frame[index] = data_frame[index] * 1

    def create_featurehasher(self, feature, n_features, prefix):
        '''
        Realiza una codificación FeatureHasher a partir de los valores de la columna de un DataFrame.
        
        Parametros:
        * feature: los valores de la columna de un DataFrame
        * n_features: valor del argumento n_features de FeatureHasher
        * prefix: prefijo de las columnas del DataFrame con el resultado de la codificación FeatureHasher
        Resultado:
        * El DataFrame con el resultado de la codificación FeatureHasher
        '''
        hasher = FeatureHasher(n_features = n_features, input_type='string')
        f = hasher.transform(feature)
        columns = [prefix + "_" + str(n + 1) for n in range(n_features)]
        hasher_df = pd.DataFrame(data=f.toarray(), columns=columns)

        return hasher_df

class IdealistaML:
    '''
    Clase con métodos para hacer ciertos análisis para la creación del modelo de Machine Learning del dataset de Idealista.
    '''

    def __init__(self, debug = False):
        '''
        Constructor
        Input:
            debug: si vale True muestra mensajes de debug
        '''
        self.__debug = debug

    def correlation_analysis(self, data, show_heatmap = False, figsize=None):
        '''
        Realiza un análisis de correlación sobre un DataFrame.
        Opcionalmente muestra una gráfica heatmap del resultado del análisis de correlación.
        
        Parametros:
        * data: DataFrame sobre el que se realiza un análisis de correlación
        * show_heatmap: si vale True muestra una gráfica heatmap del resultado del análisis de correlación
        * figsize: tamaño de la gráfica heatmap
        * prefix: prefijo de las columnas del DataFrame con el resultado de la codificación FeatureHasher
        Resultado:
        * El resultado del análisis de correlación
        '''
        correlation = data.corr()

        mask = np.zeros_like(correlation, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=figsize)

        cmap = sns.diverging_palette(180, 20, as_cmap=True)
        sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin =-1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

        plt.show()
        
        return correlation