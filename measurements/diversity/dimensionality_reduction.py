from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
import time
import matplotlib.pyplot as plt
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
import os

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
def get_pca_projection(features, n_components=2, should_fit=True, evorun_dir='', plot_variance_ratio=False, components_list=[]):
    global pca
    global scaler
    if should_fit:
        print('Fitting PCA model...')
        # if components_list has been set, use the default PCA constructor, which will use all components, and then use the components_list to select the components
        if len(components_list) > 0:
            pca = PCA()
        else: # otherwise use n_components
            pca = PCA(n_components=n_components)
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

    scaled = scaler.transform(transformed)

    # pick the components indexto to in components_list
    if len(components_list) > 0:
        scaled = scaled[:, components_list]

    return scaled

def get_umap_projection(features, n_components=2, should_fit=True, evorun_dir=''):
    umap = ParametricUMAP(
        n_components=n_components,
        # autoencoder_loss = True, # TODO: how to use this? https://umap-learn.readthedocs.io/en/latest/parametric_umap.html#autoencoding-umap
    )
    if should_fit:
        print('Fitting UMAP model...')
        umap.fit(features)

        umap_model_path = evorun_dir + 'umap_model'
        print('Saving UMAP model to disk...', umap_model_path)
        print('os.path.dirname(umap_model_path)', os.path.dirname(umap_model_path))
        if not os.path.exists(umap_model_path):
            print('Creating directory for UMAP model...', os.path.dirname(umap_model_path))
            os.makedirs(umap_model_path)
        umap.save(umap_model_path)
    else:
        # assume that the UMAP model has been saved to disk - TODO: throw an error if it hasn't?
        umap = load_ParametricUMAP(evorun_dir + 'umap_model')

    transformed = umap.transform(features)
    # transformed = umap.fit_transform(features)

    return transformed

# TODO autoencoder, t-SNE ?