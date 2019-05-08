from sklearn.decomposition import PCA
import mca

from model import visualize
import pandas as pd
import numpy as np


def select_features(X_train, X_test, y_train, y_test, threshold):
    global FEATURE_SELECTION_METHOD
    if(FEATURE_SELECTION_METHOD == "PCA"):
        pca = PCA()
        X_train = pca.fit_transform(X_train)  
        X_test = pca.transform(X_test)
        n_pca = get_count_of_selected_principle_components(pca, threshold)
        print("First " + str(n_pca) + " PCAs selected.")
        pca = PCA(n_components=n_pca)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        # visualize_pca(X_train, y_train)
    else : 
        ca = mca.MCA()
        # visualize_pca(X_train, y_train)
    return X_train, X_test, y_train, y_test


def get_count_of_selected_principle_components(pca, threshold):
    explained_variance = pca.explained_variance_ratio_
    print(explained_variance)
    # visualize.show_cumulative_sum_of_explained_variance(explained_variance)
    sum_of_variances = 0
    count_of_principle_components = 0
    for variance_ratio in explained_variance:
        if(sum_of_variances < threshold):
            sum_of_variances += variance_ratio
            count_of_principle_components += 1
        else:
            break
    
    return count_of_principle_components


def visualize_pca(X_train, y_train):
    pca = 1
    principal_df = pd.DataFrame(data=X_train[:, pca - 1:pca + 1:1], columns=['PC 1', 'PC 2'])
    X_train_df.drop('symboling', axis=1, inplace=True)
    y_train_df = pd.DataFrame(data=y_train, columns=['symboling'])
    final_df = pd.concat([principal_df, y_train_df], axis=1)
    #visualize.show_pca(final_df)
