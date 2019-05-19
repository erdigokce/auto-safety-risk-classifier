import logging

from sklearn.preprocessing import StandardScaler

import pandas as pd
from preprocess.extractor import PgiInsAutoClsFeatureExtractor


class PgiInsAutoClsPreprocessor:

    def preprocess(self, json_response):
        autos_dataframe_filled_missing_values = self.extract_features(json_response)
        # normalize values
        normalized_autos_dataframe = self._normalize(autos_dataframe_filled_missing_values)
        return normalized_autos_dataframe.drop('symboling', 1), normalized_autos_dataframe['symboling']
    
    def extract_features(self, json_response):
        autos_dataframe = pd.DataFrame(json_response)
        # drop manual fields
        self._drop_unnecessary_fields(autos_dataframe)
        # extract categorical values to numerics
        feature_extractor = PgiInsAutoClsFeatureExtractor(autos_dataframe)
        extended_autos_dataframe = feature_extractor.extract_features()
        # fill missing values
        return self.fill_missing_values(extended_autos_dataframe)
    
    def _normalize(self, df):
        df_symboling = df['symboling']
        df.drop('symboling', axis=1, inplace=True)
        logging.debug(df.dtypes)
        features = df.columns
        logging.info('Features to be normalized : \n %s', features)
        sc = StandardScaler()
        normalized_df = pd.DataFrame(sc.fit_transform(df), columns=features)
        symbolizing_df = pd.DataFrame(df_symboling, columns=['symboling'])
        final_df = pd.concat([normalized_df, symbolizing_df], axis=1)
        return final_df
    
    def fill_missing_values(self, df):
        return df.fillna(df.mean())

    def _drop_unnecessary_fields(self, df):
        df.drop('id', axis=1, inplace=True)
