from sklearn.decomposition import PCA

from model import visualize
import pandas as pd


def select_features(train_df):
    features = train_df.columns
    x = train_df.loc[:, features].values
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)
    principal_df = pd.DataFrame(data=principal_components, columns=['principal component 1', 'principal component 2'])
    final_df = pd.concat([principal_df, train_df[['symboling']]], axis=1)
    visualize.show_pca(final_df)
