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
    start_time = time.time()
    # fetch data
    json_response = integration.get_number_of_samples(10000)
    
    # preprocess : extract features and normalize.
    X, y = preprocess.preprocess(json_response)
    
    n_folds = 10
    kf = KFold(n_splits=n_folds)
    print(kf)
    sum_accuracy = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # feature selection
        X_train, X_test, y_train, y_test = dimensionality_reduction.select_features(X_train, X_test, y_train, y_test)
        
        # y_pred = _perform_predictions(X_train, X_test, y_train)
        
        cm = confusion_matrix(y_test, y_pred)  
        print(cm)
        
        accuracy = accuracy_score(y_test, y_pred) * 100
        sum_accuracy += accuracy
        print('Accuracy : ', str(accuracy), '%')
    print('-------------------------------------')
    avg_accuracy = sum_accuracy / n_folds
    print('Final Accuracy : ', avg_accuracy, '%')
    _plot_correlation_matrix(X, y)
    print('Total duration : ', time.time() - start_time, ' seconds')
    '''
    # predict one - begin
    input_vector = [235, 5, 2, 1, 4, 3, 2, 2, 280, 474, 174, 1330, 7, 4, 1997, 6, 87, 84, 22, 72, 4600, 16, 250, 14.925, 21000, '?']
    print(classifier_calculator.predict(input_vector))
    # predict one - end
    '''


def _perform_predictions(X_train, X_test, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    # Predicting the Test set results
    return model.predict(X_test)


def _plot_correlation_matrix(X, y):
    visualize.show_heatmap_of_correlation_matrix(pd.concat([X, y], axis=1))


main()
