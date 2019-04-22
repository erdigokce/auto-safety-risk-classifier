import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess(json_response):
    observations = pd.DataFrame(json_response)
    gle = LabelEncoder()
    genre_labels = gle.fit_transform(observations['make'])
    genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
    print(genre_labels)
    extended_observations = _extract_features(observations)
    normalized_observations = _normalize(extended_observations)
    return normalized_observations


def _extract_features(observations):
    print("extract features")

    
def _normalize(observations):
    print("_normalize")
