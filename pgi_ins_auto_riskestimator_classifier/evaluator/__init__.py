import logging

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from classifier import PgiInsAutoClsClassifier
from dimensionality_reduction import PgiInsAutoClsFeatureSelector


class PgiInsAutoClsEvaluator:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def perform_evaluation(self):
        threshold = 1.00
        accuracy = self._evaluate_model(threshold)
        while(accuracy < 70 and threshold > .85):
            threshold = threshold - .05
            accuracy = self._evaluate_model(threshold)
        logging.info('Final Accuracy : %.2f %%, Threshold : %.2f', accuracy, threshold)

    def _evaluate_model(self, threshold):
        n_folds = 10
        kf = KFold(n_splits=n_folds)
        logging.info('%s', kf)
        sum_accuracy = 0
        for train_index, test_index in kf.split(self.x):
            x_train, x_test = self.x.iloc[train_index], self.x.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            
            # feature selection
            feature_selector = PgiInsAutoClsFeatureSelector()
            x_train, x_test, y_train, y_test = feature_selector.select_features_for_evaluation(x_train, x_test, y_train, y_test, threshold)
            
            classifier = PgiInsAutoClsClassifier(x_train, y_train, x_test)
            y_pred = classifier.perform_predictions()
            
            cm = confusion_matrix(y_test, y_pred)  
            logging.info('Confusion matrix : \n %s', cm)
            
            accuracy = accuracy_score(y_test, y_pred) * 100
            sum_accuracy += accuracy
        avg_accuracy = sum_accuracy / n_folds
        return avg_accuracy
