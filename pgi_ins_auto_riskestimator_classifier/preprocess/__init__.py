import logging

from sklearn.preprocessing import StandardScaler

import pandas as pd
from extractor import PgiInsAutoClsFeatureExtractor


class PgiInsAutoClsPreprocessor:

    def __init__(self):
        self.logger = logging.getLogger('pgiInsAreClassifierLogger')

    def preprocess(self, json_response):
        autos_dataframe = pd.DataFrame(json_response)
        # fill missing values
        fixed_autos_dataframe = self.fill_missing_values(autos_dataframe)
        # drop manual fields
        self._drop_unnecessary_fields(fixed_autos_dataframe)
        # extract categorical values to numerics
        autos_dataframe_filled_missing_values = self.extract_features(fixed_autos_dataframe)
        self.logger.info('Feature count is %d after extraction (including symbolizing)', len(autos_dataframe_filled_missing_values.columns))
        # normalize values
        normalized_autos_dataframe = self._normalize(autos_dataframe_filled_missing_values)
        return normalized_autos_dataframe.drop('symboling', 1), normalized_autos_dataframe['symboling']
    
    def extract_features(self, df):
        feature_extractor = PgiInsAutoClsFeatureExtractor(df)
        return feature_extractor.extract_features()
    
    def _normalize(self, df):
        df_symboling = df['symboling']
        df.drop('symboling', axis=1, inplace=True)
        self.logger.debug(df.dtypes)
        features = df.columns
        self.logger.info('Features to be normalized : \n %s', features)
        sc = StandardScaler()
        normalized_df = pd.DataFrame(sc.fit_transform(df), columns=features)
        symbolizing_df = pd.DataFrame(df_symboling, columns=['symboling'])
        final_df = pd.concat([normalized_df, symbolizing_df], axis=1)
        return final_df
    
    def fill_missing_values(self, df):
        return df.fillna(df.mean())

    def _drop_unnecessary_fields(self, df):
        df.drop('id', axis=1, inplace=True)
