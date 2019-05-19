import logging

from sklearn.naive_bayes import GaussianNB


class PgiInsAutoClsClassifier:

    def __init__(self, x_train, y_train, x_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
    
    def perform_predictions(self):
        model = GaussianNB()
        model.fit(self.x_train, self.y_train)
        return model.predict(self.x_test)
