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
    Clase que permite obtener información de Idealista a partir de su API.
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
        # 'parkingSpace': {'hasParkingSpace': True, 'isParkingSpaceIncludedInPrice': True}
        hasparkingspace = np.nan
        if pd.notnull(parking_space):
            hasparkingspace = parking_space['hasParkingSpace']
        
        return hasparkingspace
        
    def __get_is_parkingspace_inc_inprice(self, parking_space):
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
        self.__token = token
        
    def search(self, center, country='es', numPage='1', maxItems='50', distance='1000', propertyType='homes', operation='sale'):
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
        with open(file, 'w') as outfile:
            json.dump(result["elementList"], outfile)
    
    def read_json(self, file, type='frame'):
        '''
        Parametros:
            type: 'frame', 'series'
        '''
        return pd.read_json(file, orient='records', typ=type)
    
    def read_jsons(self, directory='', type='frame'):
        '''
        Parametros:
            type: 'frame', 'series'
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
        data_frame["subTypology"] = data_frame["detailedType"].apply(lambda x: x.get('subTypology', ''))
        # 'parkingSpace': {'hasParkingSpace': True, 'isParkingSpaceIncludedInPrice': True}
        data_frame["hasParkingSpace"] = data_frame["parkingSpace"].apply(self.__get_has_parkingspace)
        data_frame["isParkingSpaceIncludedInPrice"] = data_frame["parkingSpace"].apply(self.__get_is_parkingspace_inc_inprice)
        #data_frame.drop(columns = ["thumbnail","externalReference","numPhotos","hasVideo","has3DTour","has360","detailedType","parkingSpace"], inplace = True)
        # Se eliminan estas 2 columnas porque se han convertido en las nuevas columnas "subTypology", "hasParkingSpace" y "isParkingSpaceIncludedInPrice"
        data_frame.drop(columns = ["detailedType","parkingSpace"], inplace = True)

class IdealistaFeatureEngineering:

    def __init__(self, debug = False):
        '''
        Constructor
        Input:
            debug: si vale True muestra mensajes de debug
        '''
        self.__debug = debug
    
    
    def nan_analysis(self, dataframe, display_analysis = True):
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
        for index, value in data_frame.dtypes.items():
            if str(value) == 'bool':
                data_frame[index] = data_frame[index] * 1

    def create_featurehasher(self, feature, n_features, prefix):
        hasher = FeatureHasher(n_features = n_features, input_type='string')
        f = hasher.transform(feature)
        columns = [prefix + "_" + str(n + 1) for n in range(n_features)]
        hasher_df = pd.DataFrame(data=f.toarray(), columns=columns)

        return hasher_df

class IdealistaML:

    def __init__(self, debug = False):
        '''
        Constructor
        Input:
            debug: si vale True muestra mensajes de debug
        '''
        self.__debug = debug

    def correlation_analysis(self, data, show_heatmap = False, figsize=None):
        correlation = data.corr()

        mask = np.zeros_like(correlation, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

        f, ax = plt.subplots(figsize=figsize)

        cmap = sns.diverging_palette(180, 20, as_cmap=True)
        sns.heatmap(correlation, mask=mask, cmap=cmap, vmax=1, vmin =-1, center=0,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

        plt.show()
        
        return correlation
    
    def filter_correlation_with_target(self, correlation, correlation_filter):
        return None