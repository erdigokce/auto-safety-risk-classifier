from os import path
import logging
import logging.config
import sys
import time

from classifier import PgiInsAutoClsClassifier
from dimensionality_reduction import PgiInsAutoClsFeatureSelector
from evaluator import PgiInsAutoClsEvaluator
import integration
from visualizer import PgiInsAutoClsVisualizer
from preprocess import PgiInsAutoClsPreprocessor


def main():
    if(len(sys.argv) > 2):
        raise Exception('More than {} parameter is not allowed! Parameters : {}', 2, sys.argv)
    # fetch data
    observation_count = int(sys.argv[1])
    json_response = integration.get_number_of_samples(observation_count)
    
    # preprocess : extract features and normalize.
    logger.info('Preprocess step has begun.')
    preprocessor = PgiInsAutoClsPreprocessor()
    x, y = preprocessor.preprocess(json_response)
    logger.info('Preprocess step has been finished with x length %d and y length %d.', len(x), len(y))
    
    PgiInsAutoClsEvaluator(x, y).perform_evaluation()
    
    # visualize.show_heatmap_of_correlation_matrix(pd.concat([x, y], axis=1))


log_config_path = path.join(path.dirname(path.abspath(__file__)), 'logging.ini')
logging.config.fileConfig(fname=log_config_path, disable_existing_loggers=False)
logger = logging.getLogger('pgiInsAreClassifierLogger')
start_time = time.time()
visualize = PgiInsAutoClsVisualizer()
main()
logger.info('Total duration : %.2f seconds', time.time() - start_time)
