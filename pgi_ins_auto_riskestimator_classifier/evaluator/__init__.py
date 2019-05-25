import logging

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

from classifier import PgiInsAutoClsClassifier
from dimensionality_reduction import PgiInsAutoClsFeatureSelector
from config import application_config


class PgiInsAutoClsEvaluator:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.logger = logging.getLogger('pgiInsAreClassifierLogger')

    def perform_evaluation(self):
        threshold = 1
        threshold_selected = .70
        accuracy_posterior = 0
        accuracy_prior = self._evaluate_model(threshold)
        if application_config['THRESHOLD_OPTIMIZER'] == 1 :
            while(threshold > .70):
                if(accuracy_posterior > accuracy_prior):
                    accuracy_prior = accuracy_posterior
                    threshold_selected = threshold
                threshold = threshold - .025
                accuracy_posterior = self._evaluate_model(threshold)
        else :
            accuracy_posterior = self._evaluate_model(threshold)
        self.logger.info('Final Accuracy : %.2f %%, Threshold : %.2f', accuracy_posterior, threshold_selected)

    def _evaluate_model(self, threshold):
        n_folds = 10
        kf = KFold(n_splits=n_folds)
        self.logger.info('%s', kf)
        sum_accuracy = 0
        for train_index, test_index in kf.split(self.x):
            x_train, x_test = self.x.iloc[train_index], self.x.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            
            # feature selection
            feature_selector = PgiInsAutoClsFeatureSelector()
            x_train, x_test, y_train, y_test = feature_selector.select_features_for_evaluation(x_train, x_test, y_train, y_test, threshold)
            
            # classification
            classifier = PgiInsAutoClsClassifier()
            classifier.fit(x_train, y_train)
            y_pred = classifier.perform_predictions(x_test)
            
            cm = confusion_matrix(y_test, y_pred)  
            self.logger.info('Confusion matrix : \n %s', cm)
            
            accuracy = accuracy_score(y_test, y_pred) * 100
            sum_accuracy += accuracy
        avg_accuracy = sum_accuracy / n_folds
        return avg_accuracy
