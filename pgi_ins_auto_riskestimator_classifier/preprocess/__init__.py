import integration
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder


def preprocess(json_response):
    observations = pd.DataFrame(json_response)
    extended_observations = _extract_features(observations)
    print(extended_observations)
    normalized_observations = _normalize(extended_observations)
    return normalized_observations


def _extract_features(observations):
    observations = _extract_feature_make(observations)
    observations = _extract_feature_fuel_type(observations)
    observations = _extract_feature_aspiration(observations)
    observations = _extract_feature_number_of_doors(observations)
    observations = _extract_feature_body_style(observations)
    observations = _extract_feature_drive_wheels(observations)
    observations = _extract_feature_engine_locations(observations)
    observations = _extract_feature_engine_type(observations)
    observations = _extract_feature_number_of_cylinders(observations)
    observations = _extract_feature_fuel_system(observations)
    return observations


def _extract_feature_make(observations):
    fh = FeatureHasher(n_features=11, input_type='string')
    hashed_features = fh.fit_transform(observations['make'])
    hashed_features = hashed_features.toarray()
    observations = observations.drop('make', axis=1)
    return pd.concat([observations, pd.DataFrame(hashed_features)], axis=1)


def _extract_feature_fuel_type(observations):
    ohe = OneHotEncoder()
    fitted_feature = ohe.fit_transform(observations[['fuelType']])
    fitted_feature_arr = fitted_feature.toarray()
    observations = observations.drop('fuelType', axis=1)
    return pd.concat([observations, pd.DataFrame(fitted_feature_arr, columns=['diesel', 'gas'])], axis=1)


def _extract_feature_aspiration(observations):
    ohe = OneHotEncoder()
    fitted_feature = ohe.fit_transform(observations[['aspiration']])
    fitted_feature_arr = fitted_feature.toarray()
    observations = observations.drop('aspiration', axis=1)
    return pd.concat([observations, pd.DataFrame(fitted_feature_arr, columns=['std', 'turbo'])], axis=1)


def _extract_feature_number_of_doors(observations):
    ordinal_map = integration.get_number_of_doors_map()
    observations['numberOfDoors'] = observations['numberOfDoors'].map(ordinal_map)
    return observations


def _extract_feature_body_style(observations):
    fh = FeatureHasher(n_features=5, input_type='string')
    hashed_features = fh.fit_transform(observations['bodyStyle'])
    hashed_features = hashed_features.toarray()
    observations = observations.drop('bodyStyle', axis=1)
    return pd.concat([observations, pd.DataFrame(hashed_features)], axis=1)


def _extract_feature_drive_wheels(observations):
    ohe = OneHotEncoder()
    fitted_feature = ohe.fit_transform(observations[['driveWheels']])
    fitted_feature_arr = fitted_feature.toarray()
    observations = observations.drop('driveWheels', axis=1)
    return pd.concat([observations, pd.DataFrame(fitted_feature_arr, columns=['4wd', 'fwd', 'rwd'])], axis=1)


def _extract_feature_engine_locations(observations):
    ohe = OneHotEncoder()
    fitted_feature = ohe.fit_transform(observations[['engineLocations']])
    fitted_feature_arr = fitted_feature.toarray()
    observations = observations.drop('engineLocations', axis=1)
    return pd.concat([observations, pd.DataFrame(fitted_feature_arr, columns=['front', 'rear'])], axis=1)


def _extract_feature_engine_type(observations):
    fh = FeatureHasher(n_features=7, input_type='string')
    hashed_features = fh.fit_transform(observations['engineType'])
    hashed_features = hashed_features.toarray()
    pd.concat([observations[['engineType']], pd.DataFrame(hashed_features)], axis=1)
    return observations


def _extract_feature_number_of_cylinders(observations):
    ordinal_map = integration.get_number_of_cylinders_map()
    observations['numberOfCylinders'] = observations['numberOfCylinders'].map(ordinal_map)
    return observations


def _extract_feature_fuel_system(observations):
    fh = FeatureHasher(n_features=8, input_type='string')
    hashed_features = fh.fit_transform(observations['fuelSystem'])
    hashed_features = hashed_features.toarray()
    pd.concat([observations[['fuelSystem']], pd.DataFrame(hashed_features)], axis=1)
    return observations


def _normalize(observations):
    print("_normalize")
