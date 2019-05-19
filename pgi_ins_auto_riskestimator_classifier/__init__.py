import logging
import logging.config
import sys
import time

from classifier import PgiInsAutoClsClassifier
from dimensionality_reduction import PgiInsAutoClsFeatureSelector
from evaluator import PgiInsAutoClsEvaluator
import integration
from model import visualize
from preprocess import PgiInsAutoClsPreprocessor


def main():
    if(len(sys.argv) > 2):
        raise Exception('More than {} parameter is not allowed! Parameters : {}', 2, sys.argv)
    # fetch data
    observation_count = int(sys.argv[1])
    json_response = integration.get_number_of_samples(observation_count)
    
    # preprocess : extract features and normalize.
    logging.info('Preprocess step has begun.')
    preprocessor = PgiInsAutoClsPreprocessor()
    x, y = preprocessor.preprocess(json_response)
    logging.info('Preprocess step has been finished.')
    
    PgiInsAutoClsEvaluator(x, y).perform_evaluation()
    
    # classify_one(x, y)
    # visualize.show_heatmap_of_correlation_matrix(pd.concat([x, y], axis=1))


def classify_one(x, y):
    logging.info('Gaussian Naive Bayes classification has begun.')
    json_mercedes = [{"id":1, "normalizedLosses":90, "make":"mercedes-benz", "fuelType":"diesel", "aspiration":"std", "numberOfDoors":"four", "bodyStyle":"sedan", "driveWheels":"rwd", "engineLocations":"front", "wheelBase":280, "length":474, "width":174, "height":142.5, "curbWeight":1330, "engineType":"ohc", "numberOfCylinders":"four", "engineSize":1997, "fuelSystem":"spdi", "bore":87, "stroke":84, "compressionRatio":22, "horsepower":72, "peakRpm":4600, "cityMpg":12, "highwayMpg":14.925, "price":21000, "symboling":"?"}]
    preprocessor = PgiInsAutoClsPreprocessor()
    x_test = preprocessor.extract_features(json_mercedes)
    x_test.drop('symboling', axis=1, inplace=True)
    feature_selector = PgiInsAutoClsFeatureSelector()
    threshold = 0.85
    x, y = feature_selector.select_features(x, y, threshold)
    logging.debug('x : %s', x)
    logging.debug('test : %s', x_test)
    classifier = PgiInsAutoClsClassifier(x, y, x_test)
    logging.info('Prediction for Mercedes 200D w124: %s', classifier.perform_predictions())
    logging.info('Gaussian Naive Bayes classification has finished.')


logging.config.fileConfig(fname='logging.ini', disable_existing_loggers=False)
logger = logging.getLogger('pgiInsAreClassifierLogger')
start_time = time.time()
main()
logger.info('Total duration : %.2f seconds', time.time() - start_time)
