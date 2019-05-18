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


def _evaluate_model(x, y, threshold):
    n_folds = 10
    kf = KFold(n_splits=n_folds)
    print(kf)
    sum_accuracy = 0
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # feature selection
        x_train, x_test, y_train, y_test = dimensionality_reduction.select_features(x_train, x_test, y_train, y_test, threshold)
        
        y_pred = _perform_predictions(x_train, x_test, y_train)
        
        cm = confusion_matrix(y_test, y_pred)  
        print(cm)
        
        accuracy = accuracy_score(y_test, y_pred) * 100
        sum_accuracy += accuracy
    avg_accuracy = sum_accuracy / n_folds
    return avg_accuracy


def _perform_predictions(x_train, x_test, y_train):
    model = GaussianNB()
    model.fit(x_train, y_train)
    # Predicting the Test set results
    return model.predict(x_test)


def _plot_correlation_matrix(x, y):
    visualize.show_heatmap_of_correlation_matrix(pd.concat([x, y], axis=1))


start_time = time.time()
main()
print('Total duration : ', time.time() - start_time, ' seconds')
