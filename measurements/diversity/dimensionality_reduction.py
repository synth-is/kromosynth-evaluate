from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
import time
import matplotlib.pyplot as plt

def plot_expected_variance_ratio(pca):
    print('Printing explained variance ratio...')

    exp_var_pca = pca.explained_variance_ratio_
    cum_sum_eigenvalues = exp_var_pca.cumsum()

    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid',label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    # save the plot to disk
    plt.savefig(f'/tmp/{time.time()}_pca_explained_variance_ratio.png')

    plt.clf()

pca = None
scaler = None
def get_pca_projection(features, n_components=2, should_fit=True, evorun_dir='', plot_variance_ratio=False):
    global pca
    global scaler
    if should_fit:
        print('Fitting PCA model...')
        pca = PCA(n_components=n_components)
        # pca = PCA(n_components=14)
        pca.fit(features)
        pkl.dump(pca, open(evorun_dir + 'pca_model.pkl', 'wb'))
    else:
        # check if pca variable is set
        if pca is None:
            print('Loading PCA model...')
            # assume that the PCA model has been saved to disk - TODO: throw an error if it hasn't?
            pca = pkl.load(open(evorun_dir + 'pca_model.pkl', 'rb'))

    if plot_variance_ratio:
        plot_expected_variance_ratio(pca)

    transformed = pca.transform(features)

    if should_fit:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(transformed)
        pkl.dump(scaler, open(evorun_dir + 'scaler.pkl', 'wb'))
    else:
        # check if scaler variable is set
        if scaler is None:
            print('Loading scaler...')
            # assume that the scaler has been saved to disk - TODO: throw an error if it hasn't?
            scaler = pkl.load(open(evorun_dir + 'scaler.pkl', 'rb'))

    return scaler.transform(transformed)


# TODO autoencoder, t-SNE, UMAP