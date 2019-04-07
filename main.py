import numpy as np
import integration
import summerize
import utils

json_response = integration.get_number_of_samples(5000)
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
train_set, test_set = utils.split_dataset(np.array(observations), 0.90)

print('Split {0} rows into train with {1} and test with {2}').format(len(observations), len(train_set), len(test_set))

print('Summerize By Class : {0}'.format(summerize.summarize_by_class(train_set)))
