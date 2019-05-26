import json
import logging
import logging.config
from os import path, getcwd
import sys
import time

from application.classifier import AutosClassifier
from application.config import application as app
from application.dimensionality_reduction import AutosFeatureSelector
from application.evaluator import AutosEvaluator
from application.preprocess import AutosPreprocessor
import pandas as pd


def main():
    print('WARNING: Current working directory is ', getcwd(), '. If you get "formatter" error be sure logging.ini file is existing in your current working directory!')
    log_config_path = path.join(getcwd(), 'logging.ini')
    logging.config.fileConfig(fname=log_config_path, disable_existing_loggers=False)
    logger = logging.getLogger('pgiInsAreClassifierLogger')
    pd.set_option('display.max_columns', 75)
    start_time = time.time()
    run()
    logger.info('Total duration : %.2f seconds', time.time() - start_time)


def run():
    # fetch data
    filename = path.join(getcwd(), 'data.json')
    with open(filename, 'r') as f:
        json_response = json.load(f)
    
    logger = logging.getLogger('pgiInsAreClassifierLogger')
    # preprocess : extract features and normalize.
    logger.info('Preprocess step has begun.')
    preprocessor = AutosPreprocessor()
    x, y = preprocessor.preprocess(json_response)
    logger.info('Preprocess step has been finished with x length %d and y length %d.', len(x), len(y))
    if app['THRESHOLD_OPTIMIZER'] == 0 :
        # feature selection
        logger.info('Feature Selection step has begun.')
        feature_selector = AutosFeatureSelector()
        x, y = feature_selector.select_features(x, y)
        logger.info('Feature Selection step has been finished.')
    # evaluation
    AutosEvaluator(x, y).perform_evaluation()
