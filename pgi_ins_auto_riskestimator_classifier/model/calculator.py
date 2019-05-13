import math


class PgiInsAutoClsCalculator:
    _summaries = {}
    _test_set = {}
    _predictions = []
    
    def __init__(self, summaries, test_set):
        self._summaries = summaries
        self._test_set = test_set

    def calculate_probability(self, x, mean, stddev):
        exponential = math.exp((-1 * math.pow(x - mean, 2)) / (2 * pow(stddev, 2)))
        return exponential / math.sqrt(2 * math.pi * pow(stddev, 2))
    
    def calculate_class_probabilities(self, input_vector):
        probabilities = {}
        for class_value, class_summaries in self._summaries.items():
            probabilities[class_value] = 1
            for i in range(len(class_summaries)):
                mean, stddev, min, max = class_summaries[i]
                x = input_vector[i]
                probabilities[class_value] *= self.calculate_probability(x, mean, stddev)
        print('calculate_class_probabilities : ' + str(probabilities))
        return probabilities
    
    def predict(self, input_vector):
        probabilities = self.calculate_class_probabilities(input_vector)
        best_label, best_prob = None, -1
        for class_value, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = class_value
        return best_label
    
    def get_predictions(self):
        for i in range(len(self._test_set)):
            result = self.predict(self._test_set[i])
            self._predictions.append(result)
        print('get_predictions : ' + str(self._predictions))
    
    def get_accuracy(self):
        correct = 0
        for x in range(len(self._test_set)):
            if self._test_set[x][-1] == self._predictions[x]:
                correct += 1
        return (correct / float(len(self._test_set))) * 100.0
