import math


class PgiInsClsCalculator:
    summaries = {}
    test_set = {}
    predictions = []
    
    def __init__(self, summaries, test_set):
        self.summaries = summaries
        self.test_set = test_set

    def calculate_probability(x, mean, stddev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stddev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stddev)) * exponent
    
    def calculate_class_probabilities(input_vector):
        probabilities = {}
        for class_value, class_summaries in self.summaries.items():
            probabilities[class_value] = 1
            for i in range(len(class_summaries)):
                mean, stddev = class_summaries[i]
                x = input_vector[i]
                probabilities[class_value] *= calculate_probability(x, mean, stddev)
        print('calculate_class_probabilities : ' + str(probabilities))
        return probabilities
    
    def predict(input_vector):
        probabilities = calculate_class_probabilities(self.summaries, input_vector)
        best_label, best_prob = None, -1
        for classValue, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = classValue
        return best_label
    
    def get_predictions():
        for i in range(len(self.test_set)):
            result = predict(self.test_set[i])
            self.predictions.append(result)
        print('get_predictions : ' + str(self.predictions))
    
    def get_accuracy():
        correct = 0
        for x in range(len(self.test_set)):
            if self.test_set[x][-1] == self.predictions[x]:
                correct += 1
        return (correct / float(len(self.test_set))) * 100.0
