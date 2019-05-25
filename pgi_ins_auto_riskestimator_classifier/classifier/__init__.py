import logging
import math

from sklearn.naive_bayes import GaussianNB

import pandas as pd
import numpy as np


class PgiInsAutoClsClassifier:
    _summary = []

    def __init__(self):
        self.logger = logging.getLogger('pgiInsAreClassifierLogger')
    
    def fit(self, x_train, y_train):
        print(np.concatenate((x_train, y_train), axis=0))
        self._summary = self.summarize_by_class(df)
        
    def perform_predictions(self, x_test):
        predictions = []
        for i in range(len(x_test)):
            result = self.predict(x_test[i])
            predictions.append(result)
        self.logger.info('Get Predictions : %s', str(predictions))
        return predictions
        
    def predict(self, input_vector):
        probabilities = self.calculate_class_probabilities(input_vector)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label
    
    def calculate_class_probabilities(self, input_vector):
        probabilities = {}
        for class_value, class_summaries in self._summary.items():
            probabilities[class_value] = 1
            for i in range(len(class_summaries)):
                mean, stddev = class_summaries[i]
                x = input_vector[i]
                probabilities[class_value] *= self.calculate_probability(x, mean, stddev)
        self.logger.info('Class Probabilities : %s', str(probabilities))
        return probabilities

    def calculate_probability(self, x, mean, stddev):
        exponential = math.exp((-1 * math.pow(x - mean, 2)) / (2 * pow(stddev, 2)))
        return exponential / math.sqrt(2 * math.pi * pow(stddev, 2))
    
    def get_accuracy(self, x_test):
        correct = 0
        predictions = self.perform_predictions(x_test)
        for x in range(len(x_test)):
            if x_test[x][-1] == predictions[x]:
                correct += 1
        return (correct / float(len(x_test))) * 100.0

    def summarize_by_class(self, dataset):
        separated = self.separate_by_class(dataset)
        summaries = {}
        for class_value, instances in separated.items():
            summaries[class_value] = self.summarize(instances)
        self.logger.info('Summarize by Class : %s', str(summaries))
        return summaries
    
    def separate_by_class(self, dataset):
        separated = {}
        for i in range(len(dataset)):
            vector = dataset[i]
            if vector[-1] not in separated:
                separated[vector[-1]] = []
            separated[vector[-1]].append(vector)
        return separated
    
    def summarize(self, dataset):
        summaries = [(self.mean(attribute), self.stddev(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]
        return summaries
    
    def mean(self, numbers):
        return sum(numbers) / float(len(numbers))
    
    def stddev(self, numbers):
        avg = self.mean(numbers)
        variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers))
        return math.sqrt(variance)
