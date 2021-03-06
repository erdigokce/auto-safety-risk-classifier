import logging

from application.config import application as app
from application.visualizer import AutosVisualizer
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class AutosFeatureSelector:

    def __init__(self):
        self.pca = PCA()
        self.logger = logging.getLogger('pgiInsAreClassifierLogger')
        self.visualize = AutosVisualizer()

    def select_features(self, x, y):
        self.x_columns = x.columns
        x = self.pca.fit_transform(x)  
        n_pca = self._get_count_of_selected_principle_components(app['CUSTOM_THRESHOLD'])
        self.logger.info("[select_features] - First %d PCAs selected.", n_pca)
        self.pca = PCA(n_components=n_pca)
        x = self.pca.fit_transform(x)
        self.logger.debug("[select_features] - Principle components : \n %s", pd.DataFrame(self.pca.components_,columns=self.x_columns))
        # self._visualize_pca(x, y, 1)
        # self._visualize_correlation(x, y)
        return x, y

    def select_features_for_evaluation(self, x_train, x_test, y_train, y_test, threshold):
        x_train = self.pca.fit_transform(x_train)  
        x_test = self.pca.transform(x_test)
        n_pca = self._get_count_of_selected_principle_components(threshold)
        self.logger.info("[select_features_for_evaluation] - First %d PCAs selected.", n_pca)
        self.pca = PCA(n_components=n_pca)
        x_train = self.pca.fit_transform(x_train)
        x_test = self.pca.transform(x_test)
        # self._visualize_pca(x_train, y_train, 1)
        # self._visualize_correlation(x_train, y_train)
        return x_train, x_test, y_train, y_test
    
    def _get_count_of_selected_principle_components(self, threshold):
        explained_variance = self.pca.explained_variance_ratio_
        # self.visualize.show_cumulative_sum_of_explained_variance(explained_variance)
        sum_of_variances = 0
        count_of_principle_components = 0
        for variance_ratio in explained_variance:
            self.logger.debug("Variance ratio is %.4f for threshold %.2f", variance_ratio, threshold)
            if(sum_of_variances < threshold):
                sum_of_variances += variance_ratio
                count_of_principle_components += 1
            else:
                break
        
        return count_of_principle_components
    
    def _visualize_pca(self, x_train, y_train, pca):
        principal_df = pd.DataFrame(data=x_train[:, pca - 1:pca + 1:1], columns=['PC 1', 'PC 2'])
        y_train_df = pd.DataFrame(data=y_train, columns=['symboling'])
        final_df = pd.concat([principal_df, y_train_df], axis=1)
        self.visualize.show_pca(final_df)
        
    def _visualize_correlation(self, x_train, y_train):
        # visualize correlation
        x_train_df = pd.DataFrame(data=x_train[:, 0: x_train.size - 1])
        y_train_df = pd.DataFrame(data=y_train, columns=['symboling'])
        self.visualize.show_heatmap_of_correlation_matrix(pd.concat([x_train_df, y_train_df], axis=1))
