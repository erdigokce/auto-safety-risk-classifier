from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix  
from sklearn.model_selection import train_test_split

import dimensionality_reduction
import integration
from model import summarize, visualize
from model.calculator import PgiInsAutoClsCalculator
from sklearn.naive_bayes import GaussianNB
import numpy as np
import pandas as pd
import preprocess
from scipy.io.arff.arffread import loadarff


def main():
    # fetch data
    # raw_data = loadarff('C:\\Users\\Erdi\\Desktop\\Books\\ML\\BNG_autos_reduced.arff')
    # json_response = pd.DataFrame(raw_data[0])
    json_response = integration.get_number_of_samples(100000)
    
    # preprocess : extract features and normalize.
    X, y = preprocess.preprocess(json_response)
    
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)  
    
    # feature selection
    X_train, X_test, y_train, y_test = dimensionality_reduction.select_features(X_train, X_test, y_train, y_test)

    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Predicting the Test set results
    y_pred = model.predict(X_test)  
    
    cm = confusion_matrix(y_test, y_pred)  
    print(cm)
    print('Accuracy : ' + str(accuracy_score(y_test, y_pred) * 100) + '%')  
    '''
    # predict one - begin
    input_vector = [235, 5, 2, 1, 4, 3, 2, 2, 280, 474, 174, 1330, 7, 4, 1997, 6, 87, 84, 22, 72, 4600, 16, 250, 14.925, 21000, '?']
    print(classifier_calculator.predict(input_vector))
    # predict one - end
    '''


main()
