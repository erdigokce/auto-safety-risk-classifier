import numpy as np
import integration
from model import calculator, summarize, visualize, utils
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt


def preprocess():
    json_response = integration.get_number_of_samples(5000)
    observations = []
    for observation_in_json in json_response:
        observations.append([observation_in_json['normalizedLosses'],  #
                             int(observation_in_json['make']),  #
                             int(observation_in_json['fuelType']),  #
                             int(observation_in_json['aspiration']),  #
                             int(observation_in_json['numberOfDoors']),  #
                             int(observation_in_json['bodyStyle']),  #
                             int(observation_in_json['driveWheels']),  #
                             int(observation_in_json['engineLocations']),  #
                             observation_in_json['wheelBase'],  #
                             observation_in_json['length'],  #
                             observation_in_json['width'],  #
                             observation_in_json['curbWeight'],  #
                             int(observation_in_json['engineType']),  #
                             int(observation_in_json['numberOfCylinders']),  #
                             observation_in_json['engineSize'],  #
                             int(observation_in_json['fuelSystem']),  #
                             observation_in_json['bore'],  #
                             observation_in_json['stroke'],  #
                             observation_in_json['compressionRatio'],  #
                             observation_in_json['horsepower'],  #
                             observation_in_json['peakRpm'],  #
                             observation_in_json['cityMpg'],  #
                             observation_in_json['highwayMpg'],  #
                             observation_in_json['price'],  #
                             int(observation_in_json['symboling'])
                             ])
    train_set, test_set = utils.split_dataset(np.array(observations), 0.67)
    print 'Split {0} rows into train with {1} and test with {2}'.format(len(observations), len(train_set),
                                                                        len(test_set))
    return train_set, test_set


def feature_selection(train_set):
    print 'To be implemented'


def main():
    train_set, test_set = preprocess()

    feature_selection(train_set)

    summaries = summarize.summarize_by_class(train_set)

    # predict one - begin
    input_vector = [235, 5, 2, 1, 4, 3, 2, 2, 280, 474, 174, 1330, 7, 4, 1997, 6, 87, 84, 22, 72, 4600, 16, 250, 14.925,
                    21000, '?']
    print calculator.predict(summaries, input_vector)
    # predict one - end

    predictions = calculator.get_predictions(summaries, test_set)

    visualize.show_predictions(test_set, predictions)

    accuracy = calculator.get_accuracy(test_set, predictions)

    print('Accuracy: {0}%').format(accuracy)


main()
