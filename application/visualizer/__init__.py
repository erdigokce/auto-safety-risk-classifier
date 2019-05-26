import logging
import math

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns


class AutosVisualizer:

    def __init__(self):
        self.logger = logging.getLogger('pgiInsAreClassifierLogger')
    
    def show_heatmap_of_correlation_matrix(self, df):
        corrmat = df.corr()
        top_corr_features = corrmat.index
        plt.figure(figsize=(50, 50))
        self.logger.debug('[show_heatmap_of_correlation_matrix] - Correlation matrix : \n %s', df[top_corr_features].corr())
        # plot heat map
        g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
        plt.show()
    
    def show_pca(self, final_df):
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('Principal Component 1', fontsize=15)
        ax.set_ylabel('Principal Component 2', fontsize=15)
        ax.set_title('2 component PCA', fontsize=20)
        targets = [-3, -2, -1, 0, 1, 2, 3]
        colors = ['green', 'green', 'green', 'grey', 'red', 'red', 'red']
        for target, color in zip(targets, colors):
            indices_to_keep = final_df['symboling'] == target
            ax.scatter(final_df.loc[indices_to_keep, 'PC 1']
                       , final_df.loc[indices_to_keep, 'PC 2']
                       , c=color
                       , s=5)
        ax.legend(targets)
        ax.grid()
        self.logger.debug('[show_pca] - Principle components : \n %s', final_df)
        plt.show()
    
    def show_cumulative_sum_of_explained_variance(self, explained_variance):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title('Cumulative sum of explained variance', fontsize=20)
        ax.set_xlabel('Variance ratio', fontsize=15)
        ax.set_ylabel('Sum of variance percentage', fontsize=15)
        ax.hist(explained_variance, 1000, density=True, histtype='step', cumulative=True, label='Empirical')
        ax.grid()
        self.logger.debug('[show_cumulative_sum_of_explained_variance] - Explained variance : \n %s', explained_variance)
        plt.show()
        
    def show_pdf_of_class(self, class_summary):
        mu = class_summary[-1][0]
        variance = class_summary[-1][1]
        sigma = math.sqrt(variance)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma))
        plt.show()
