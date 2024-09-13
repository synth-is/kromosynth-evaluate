from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pickle as pkl
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
import os
from scipy.stats import entropy

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

# TODO: should reload model

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



def create_autoencoder(input_dim, latent_dim, random_state):
    tf.random.set_seed(random_state)
    encoder = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(latent_dim, activation='linear')
    ])
    
    decoder = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(latent_dim,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(input_dim, activation='linear')
    ])
    
    return encoder, decoder

def calculate_reconstruction_loss(umap_model, features):
    reconstructed = umap_model.decoder(umap_model.transform(features)).numpy()
    return np.mean(np.square(features - reconstructed), axis=1)

def calculate_complexity(feature_vector):
    # Calculate entropy as a measure of complexity
    # hist, _ = np.histogram(features, bins=20)
    # return entropy(hist + 1e-10)  # Add small constant to avoid log(0)
    hist, _ = np.histogram(feature_vector, bins='auto', density=True)
    return entropy(hist + 1e-10)  # Add small constant to avoid log(0)

def calculate_smoothness(feature_vector):
    # Calculate the inverse of the average gradient as a measure of smoothness
    # gradients = np.diff(features, axis=1)
    # return 1 / (np.mean(np.abs(gradients)) + 1e-10)  # Add small constant to avoid division by zero
    total_variation = np.sum(np.abs(np.diff(feature_vector)))
    return 1 / (total_variation + 1e-10)  # Add small constant to avoid division by zero

def calculate_novelty_score(reconstruction_loss, max_reconstruction_error, feature_vector, max_complexity, max_smoothness, alpha=10, beta=0.5, gamma=0.5):
    normalized_loss = reconstruction_loss / max_reconstruction_error
    complexity = calculate_complexity(feature_vector)
    smoothness = calculate_smoothness(feature_vector)
    
    # Normalize complexity and smoothness
    normalized_complexity = complexity / max_complexity
    normalized_smoothness = smoothness / max_smoothness
    
    # Combine reconstruction loss, complexity penalty, and smoothness reward
    adjusted_loss = normalized_loss - beta * normalized_complexity + gamma * normalized_smoothness
    
    # Apply sigmoid to get final novelty score
    return 1 / (1 + np.exp(-alpha * (adjusted_loss - 0.5)))

def get_umap_projection(features, n_components=2, should_fit=True, evorun_dir='', random_state=42, n_neighbors=15, min_dist=0.1):
    input_dim = features.shape[1]
    
    if should_fit:
        encoder, decoder = create_autoencoder(input_dim, n_components, random_state)
        
        umap = ParametricUMAP(
            n_components=n_components,
            encoder=encoder,
            decoder=decoder,
            autoencoder_loss=True,
            loss_report_frequency=1,
            n_epochs=100,  # Adjust as needed
            batch_size=64,  # Adjust as needed
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            init='random'  # Use random initialization for better reproducibility
        )
        
        print('Fitting UMAP model...')
        umap.fit(features)

        umap_model_path = os.path.join(evorun_dir, 'umap_model')
        print('Saving UMAP model to disk...', umap_model_path)
        if not os.path.exists(umap_model_path):
            print('Creating directory for UMAP model...', os.path.dirname(umap_model_path))
            os.makedirs(umap_model_path)
        umap.save(umap_model_path)
        
        # Calculate and save the max values for normalization
        all_reconstruction_losses = calculate_reconstruction_loss(umap, features)
        max_reconstruction_error = np.max(all_reconstruction_losses)
        max_complexity = np.max([calculate_complexity(f) for f in features])
        max_smoothness = np.max([calculate_smoothness(f) for f in features])
        
        np.save(os.path.join(evorun_dir, 'max_reconstruction_error.npy'), max_reconstruction_error)
        np.save(os.path.join(evorun_dir, 'max_complexity.npy'), max_complexity)
        np.save(os.path.join(evorun_dir, 'max_smoothness.npy'), max_smoothness)
    else:
        umap_model_path = os.path.join(evorun_dir, 'umap_model')
        if not os.path.exists(umap_model_path):
            raise FileNotFoundError(f"UMAP model not found at {umap_model_path}")
        umap = load_ParametricUMAP(umap_model_path)
        max_reconstruction_error = np.load(os.path.join(evorun_dir, 'max_reconstruction_error.npy'))
        max_complexity = np.load(os.path.join(evorun_dir, 'max_complexity.npy'))
        max_smoothness = np.load(os.path.join(evorun_dir, 'max_smoothness.npy'))

    # Ensure features is 2D
    if features.ndim == 1:
        features = features.reshape(1, -1)

    transformed = umap.transform(features)
    reconstruction_losses = calculate_reconstruction_loss(umap, features)
    
    novelty_scores = np.array([
        calculate_novelty_score(loss, max_reconstruction_error, feature, max_complexity, max_smoothness)
        for loss, feature in zip(reconstruction_losses, features)
    ])

    return transformed, novelty_scores

def set_global_random_state(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)