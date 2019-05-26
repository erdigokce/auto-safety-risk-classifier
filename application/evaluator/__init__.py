import logging
import random

from application.classifier import AutosClassifier
from application.config import application as app
from application.dimensionality_reduction import AutosFeatureSelector
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


class AutosEvaluator:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.logger = logging.getLogger('pgiInsAreClassifierLogger')

    def perform_evaluation(self):
        threshold = 1
        threshold_selected = .70
        accuracy_posterior = 0
        if app['THRESHOLD_OPTIMIZER'] == 1 :
            accuracy_prior = self._evaluate_model(threshold)
            while(threshold > .70):
                if(accuracy_posterior > accuracy_prior):
                    accuracy_prior = accuracy_posterior
                    threshold_selected = threshold
                threshold = threshold - .025
                accuracy_posterior = self._evaluate_model(threshold)
        else :
            accuracy_prior = self._evaluate_model(app['CUSTOM_THRESHOLD'])
            accuracy_posterior = self._evaluate_model(app['CUSTOM_THRESHOLD'])
        self.logger.info('Final Accuracy : %.2f %%, Threshold : %.2f', accuracy_posterior, threshold_selected)

    def _evaluate_model(self, threshold):
        accuracy = 0
        if(app['EVALUATION_METHOD']['METHOD_NAME'] == 'KFold'):
            accuracy = self._evaluate_model_with_kfold(app['EVALUATION_METHOD']['METHOD_VALUE'], threshold)
        else :
            accuracy = self._evaluate_model_with_split_by_ratio(app['EVALUATION_METHOD']['METHOD_VALUE'], threshold)
        return accuracy
    
    def _evaluate_model_with_kfold(self, n_fold, threshold):
        n_folds = 10
        kf = KFold(n_splits=n_folds)
        self.logger.info('%s', kf)
        sum_accuracy = 0
        for train_index, test_index in kf.split(self.x):
            x_train, x_test = self.x.iloc[train_index], self.x.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            accuracy = self._run_model(x_train, x_test, y_train, y_test, threshold)
            sum_accuracy += accuracy
        avg_accuracy = sum_accuracy / n_folds
        return avg_accuracy

    def _evaluate_model_with_split_by_ratio(self, ratio, threshold):
        dataset = np.append(self.x, [[y] for y in self.y], axis=1)
        train_size = int(len(dataset) * ratio)
        train_set = []
        test_set = list(dataset)
        while len(train_set) < train_size:
            index = random.randrange(len(test_set))
            train_set.append(test_set.pop(index))
        y_train = [row[-1:] for row in train_set]
        x_train = [row[0:row.size - 1] for row in train_set]
        y_test = [row[-1:] for row in test_set]
        x_test = [row[0:row.size - 1] for row in test_set]
        return self._run_model(x_train, x_test, y_train, y_test, threshold)

    def _run_model(self, x_train, x_test, y_train, y_test, threshold):
        if app['THRESHOLD_OPTIMIZER'] == 1 :
            # feature selection
            self.logger.info('Feature Selection step has begun.')
            feature_selector = AutosFeatureSelector()
            x_train, x_test, y_train, y_test = feature_selector.select_features_for_evaluation(x_train, x_test, y_train, y_test, threshold)
            self.logger.info('Feature Selection step has been finished.')
        # classification
        classifier = AutosClassifier()
        classifier.fit(x_train, y_train)
        y_pred = classifier.perform_predictions(x_test)
        # matrix of expected class to classified class
        cm = confusion_matrix(y_test, y_pred)  
        self.logger.info('Confusion matrix : \n %s', cm)
        
        return accuracy_score(y_test, y_pred) * 100
