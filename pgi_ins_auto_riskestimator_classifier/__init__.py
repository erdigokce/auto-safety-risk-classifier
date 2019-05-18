import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import dimensionality_reduction
import integration
from model import visualize
import numpy as np
import pandas as pd
import preprocess


def main():
    # fetch data
    json_response = integration.get_number_of_samples(30000)
    
    # preprocess : extract features and normalize.
    X, y = preprocess.preprocess(json_response)
    
    threshold = 1.00
    accuracy = _evaluate_model(X, y, threshold)
    while(accuracy < 70 and threshold > .80):
        threshold = threshold - .025
        accuracy = _evaluate_model(X, y, threshold)
    print('Final Accuracy : ', accuracy, '%')
    # _plot_correlation_matrix(X, y)


def _evaluate_model(X, y, threshold):
    n_folds = 10
    kf = KFold(n_splits=n_folds)
    print(kf)
    sum_accuracy = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # feature selection
        X_train, X_test, y_train, y_test = dimensionality_reduction.select_features(X_train, X_test, y_train, y_test, threshold)
        
        y_pred = _perform_predictions(X_train, X_test, y_train)
        
        cm = confusion_matrix(y_test, y_pred)  
        # print(cm)
        
        accuracy = accuracy_score(y_test, y_pred) * 100
        sum_accuracy += accuracy
        # print('Accuracy : ', str(accuracy), '%')
    # print('-------------------------------------')
    avg_accuracy = sum_accuracy / n_folds
    # print('Final Accuracy : ', avg_accuracy, '%')
    return avg_accuracy


def _perform_predictions(X_train, X_test, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    # Predicting the Test set results
    return model.predict(X_test)


def _plot_correlation_matrix(X, y):
    visualize.show_heatmap_of_correlation_matrix(pd.concat([X, y], axis=1))


start_time = time.time()
main()
print('Total duration : ', time.time() - start_time, ' seconds')
