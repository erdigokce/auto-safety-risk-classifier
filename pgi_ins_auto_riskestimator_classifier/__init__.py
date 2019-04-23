import integration
import preprocess
import dimensionality_reduction
from model import summarize, visualize, utils
from model.calculator import PgiInsClsCalculator
import numpy as np


def main():
    # fetch data
    json_response = integration.get_number_of_samples(1000)
    
    # preprocess : extract features and normalize.
    observations = preprocess.preprocess(json_response)
    
    # feature selection
    observations_with_best_features = dimensionality_reduction.select_features(observations)
    
    # Split data set into training and test.
    train_set, test_set = utils.split_dataset(np.array(observations_with_best_features), 0.67)
    print('Split %d rows into train with %d and test with %d' % (len(observations_with_best_features), len(train_set), len(test_set)))

    summaries = summarize.summarize_by_class(train_set)

    classifier_calculator = PgiInsClsCalculator(summaries, test_set)

    # predict one - begin
    input_vector = [235, 5, 2, 1, 4, 3, 2, 2, 280, 474, 174, 1330, 7, 4, 1997, 6, 87, 84, 22, 72, 4600, 16, 250, 14.925, 21000, '?']
    print(classifier_calculator.predict(input_vector))
    # predict one - end

    predictions = classifier_calculator.get_predictions()

    visualize.show_predictions(test_set, predictions)

    accuracy = classifier_calculator.get_accuracy()

    print('Accuracy: %f%%' % (accuracy))


main()
