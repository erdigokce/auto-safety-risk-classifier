import matplotlib.pyplot as plt
import seaborn as sns


def show_heatmap_of_correlation_matrix(df):
    corrmat = df.corr()
    top_corr_features = corrmat.index
    plt.figure(figsize=(20, 20))
    # plot heat map
    g = sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
    plt.show()


def show_pca(final_df):
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
    plt.show()


def show_cumulative_sum_of_explained_variance(explained_variance):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_title('Cumulative sum of explained variance', fontsize=20)
    ax.hist(explained_variance, 1000, density=True, histtype='step', cumulative=True, label='Empirical')
    ax.grid()
    plt.show()
