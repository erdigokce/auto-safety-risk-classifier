import logging
import logging.config
from os import path, getcwd
import sys
import time

from application.classifier import PgiInsAutoClsClassifier
from application.config import application as app
from application.dimensionality_reduction import PgiInsAutoClsFeatureSelector
from application.evaluator import PgiInsAutoClsEvaluator
import application.integration
from application.preprocess import PgiInsAutoClsPreprocessor


def main():
    log_config_path = getcwd() + '\logging.ini'
    logging.config.fileConfig(fname=log_config_path, disable_existing_loggers=False)
    logger = logging.getLogger('pgiInsAreClassifierLogger')
    start_time = time.time()
    run()
    logger.info('Total duration : %.2f seconds', time.time() - start_time)


def run():
    if(len(sys.argv) > 2):
        raise Exception('More than {} parameter is not allowed! Parameters : {}', 2, sys.argv)
    # fetch data
    observation_count = int(sys.argv[1])
    json_response = integration.get_number_of_samples(observation_count)
    
    logger = logging.getLogger('pgiInsAreClassifierLogger')
    # preprocess : extract features and normalize.
    logger.info('Preprocess step has begun.')
    preprocessor = PgiInsAutoClsPreprocessor()
    x, y = preprocessor.preprocess(json_response)
    logger.info('Preprocess step has been finished with x length %d and y length %d.', len(x), len(y))
    if app['THRESHOLD_OPTIMIZER'] == 0 :
        # feature selection
        logger.info('Feature Selection step has begun.')
        feature_selector = PgiInsAutoClsFeatureSelector()
        x, y = feature_selector.select_features(x, y)
        logger.info('Feature Selection step has been finished.')
    # evaluation
    PgiInsAutoClsEvaluator(x, y).perform_evaluation()
