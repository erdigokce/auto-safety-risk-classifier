import pandas as pd
from preprocess.extractor import PgiInsAutoClsFeatureExtractor


def preprocess(json_response):
    autos_dataframe = pd.DataFrame(json_response)
    autos_dataframe.drop('id', axis=1, inplace=True)
    feature_extractor = PgiInsAutoClsFeatureExtractor(autos_dataframe)
    extended_autos_dataframe = feature_extractor.extract_features()
    normalized_autos_dataframe = _normalize(extended_autos_dataframe)
    return normalized_autos_dataframe


def _normalize(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    print("before normalized : \n")
    print(df.head(10))
    print("after normalized : \n")
    print(normalized_df.head(10))
    return normalized_df
