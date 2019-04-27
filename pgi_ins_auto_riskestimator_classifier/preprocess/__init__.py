import pandas as pd
from preprocess.extractor import PgiInsAutoClsFeatureExtractor
from sklearn.preprocessing import StandardScaler


def preprocess(json_response):
    autos_dataframe = pd.DataFrame(json_response)
    # autos_dataframe.drop('id', axis=1, inplace=True)
    # extract categorical values to numerics
    feature_extractor = PgiInsAutoClsFeatureExtractor(autos_dataframe)
    extended_autos_dataframe = feature_extractor.extract_features()
    # fill missing values
    autos_dataframe_filled_missing_values = fill_missing_values(extended_autos_dataframe)
    # normalize values
    normalized_autos_dataframe = _normalize(autos_dataframe_filled_missing_values)
    return normalized_autos_dataframe.drop('symboling', 1), normalized_autos_dataframe['symboling']


def _normalize(df):
    # normalized_df = (df - df.min()) / (df.max() - df.min())
    # normalized_df.drop('symboling', axis=1, inplace=True)
    # final_df = pd.concat([normalized_df, pd.DataFrame(df['symboling'], columns=['symboling'])], axis=1)
    df_symboling = df['symboling']
    df.drop('symboling', axis=1, inplace=True)
    features = df.columns
    sc = StandardScaler()
    normalized_df = pd.DataFrame(sc.fit_transform(df), columns=features)
    symbolizing_df = pd.DataFrame(df_symboling, columns=['symboling'])
    final_df = pd.concat([normalized_df, symbolizing_df], axis=1)
    return final_df


def fill_missing_values(df):
    return df.fillna(df.mean())
