from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl

pca = None
scaler = None
def get_pca_projection(features, n_components=2, should_fit=True, evorun_dir=''):
    global pca
    global scaler
    if should_fit:
        print('Fitting PCA model...')
        pca = PCA(n_components=n_components)
        pca.fit(features)
        pkl.dump(pca, open(evorun_dir + 'pca_model.pkl', 'wb'))
    else:
        # check if pca variable is set
        if pca is None:
            print('Loading PCA model...')
            # assume that the PCA model has been saved to disk - TODO: throw an error if it hasn't?
            pca = pkl.load(open(evorun_dir + 'pca_model.pkl', 'rb'))

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