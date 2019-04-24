import matplotlib.pyplot as plt


def show_pca(final_df):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1) 
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [-3, -2, -1, 0, 1, 2, 3]
    colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'purple']
    for target, color in zip(targets, colors):
        indices_to_keep = final_df['symboling'] == target
        ax.scatter(final_df.loc[indices_to_keep, 'principal component 1']
                   , final_df.loc[indices_to_keep, 'principal component 2']
                   , c=color
                   , s=2)
    ax.legend(targets)
    ax.grid()
    plt.show()
