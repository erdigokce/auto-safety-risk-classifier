from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import pandas as pd


class PgiInsAutoClsFeatureExtractor:
    _autos_dataframe = None
    
    def __init__(self, df):
        self._autos_dataframe = df
        
    def extract_features(self):
        feature_names = ['make', 'fuelType', 'aspiration', 'bodyStyle', 'driveWheels', 'engineLocations', 'engineType', 'fuelSystem', 'numberOfDoors', 'numberOfCylinders']
        # self._convert_bytes_to_strings()
        for feature_name in feature_names:
            self._extract_feature_one_hot_encoder(feature_name)
        return self._autos_dataframe
    
    def _extract_feature_one_hot_encoder(self, feature_name):
        ohe = OneHotEncoder()
        column_values = self._autos_dataframe[feature_name].unique()
        fitted_feature = ohe.fit_transform(self._autos_dataframe[[feature_name]])
        fitted_feature_arr = fitted_feature.toarray()
        self._autos_dataframe = self._autos_dataframe.drop(feature_name, axis=1)
        self._autos_dataframe = pd.concat([self._autos_dataframe, pd.DataFrame(fitted_feature_arr, columns=column_values)], axis=1)
    
    def _convert_bytes_to_strings(self):
        str_df = self._autos_dataframe.select_dtypes([np.object])
        str_df = str_df.stack().str.decode('utf-8').unstack()
        for col in str_df:
            self._autos_dataframe[col] = str_df[col]
