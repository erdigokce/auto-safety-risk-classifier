import numpy as np
import integration, summerize, utils, calculator


def main():
    json_response = integration.get_number_of_samples(20000)
    observations = []
    for observation_in_json in json_response:
        observations.append([observation_in_json['normalizedLosses'],  #
                             observation_in_json['make'],  #
                             observation_in_json['fuelType'],  #
                             observation_in_json['aspiration'],  #
                             observation_in_json['numberOfDoors'],  #
                             observation_in_json['bodyStyle'],  #
                             observation_in_json['driveWheels'],  #
                             observation_in_json['engineLocations'],  #
                             observation_in_json['wheelBase'],  #
                             observation_in_json['length'],  #
                             observation_in_json['width'],  #
                             observation_in_json['curbWeight'],  #
                             observation_in_json['engineType'],  #
                             observation_in_json['numberOfCylinders'],  #
                             observation_in_json['engineSize'],  #
                             observation_in_json['fuelSystem'],  #
                             observation_in_json['bore'],  #
                             observation_in_json['stroke'],  #
                             observation_in_json['compressionRatio'],  #
                             observation_in_json['horsepower'],  #
                             observation_in_json['peakRpm'],  #
                             observation_in_json['cityMpg'],  #
                             observation_in_json['highwayMpg'],  #
                             observation_in_json['price'],  #
                             observation_in_json['symboling']
                             ])
    train_set, test_set = utils.split_dataset(np.array(observations), 0.67)

    print 'Split {0} rows into train with {1} and test with {2}'.format(len(observations), len(train_set),
                                                                        len(test_set))

    summaries = summerize.summarize_by_class(train_set);

    print 'Summerize By Class : {0}'.format(summaries)

    input_vector = [235, 5, 2, 1, 4, 3, 2, 2, 280, 474, 174, 1330, 7, 4, 1997, 6, 87, 84, 22, 72, 4600, 16,250, 14.925, 21000, '?']
    result = calculator.predict(summaries, input_vector)

    print('Prediction: {0}').format(result)

    predictions = calculator.get_predictions(summaries, test_set)

    print 'Predictions : {0}%'.format(predictions)

    accuracy = calculator.get_accuracy(test_set, predictions)

    print('Accuracy: {0}').format(accuracy)


main()
