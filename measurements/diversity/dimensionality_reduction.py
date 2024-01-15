from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def get_pca_projection(features, n_components=2):
    pca = PCA(n_components=n_components)
    transformed = pca.fit(features).transform(features)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(transformed)
    return scaler.transform(transformed)


# TODO autoencoder, t-SNE, UMAP