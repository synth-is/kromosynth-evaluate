import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
import os
import pickle as pkl
from scipy.stats import entropy
import time

class ModelManager:
    def __init__(self, evorun_dir):
        self.evorun_dir = evorun_dir
        self.pca = None
        self.pca_autoencoder = None
        self.scaler = None
        self.umap = None
        self.max_reconstruction_error = None
        self.max_complexity = None
        self.max_smoothness = None
        self.timestamps = {}

    def save_model(self):
        os.makedirs(self.evorun_dir, exist_ok=True)
        current_time = time.time()

        if self.pca is not None:
            pca_path = os.path.join(self.evorun_dir, 'pca_model.pkl')
            pkl.dump(self.pca, open(pca_path, 'wb'))
            self.timestamps['pca'] = current_time

        if self.pca_autoencoder is not None:
            pca_ae_path = os.path.join(self.evorun_dir, 'pca_autoencoder')
            self.pca_autoencoder.save(pca_ae_path)
            self.timestamps['pca_autoencoder'] = time.time()

        if self.scaler is not None:
            scaler_path = os.path.join(self.evorun_dir, 'scaler.pkl')
            pkl.dump(self.scaler, open(scaler_path, 'wb'))
            self.timestamps['scaler'] = current_time

        if self.umap is not None:
            umap_path = os.path.join(self.evorun_dir, 'umap_model')
            self.umap.save(umap_path)
            self.timestamps['umap'] = current_time

        if self.max_reconstruction_error is not None:
            np.save(os.path.join(self.evorun_dir, 'max_reconstruction_error.npy'), self.max_reconstruction_error)
            self.timestamps['max_reconstruction_error'] = current_time

        if self.max_complexity is not None:
            np.save(os.path.join(self.evorun_dir, 'max_complexity.npy'), self.max_complexity)
            self.timestamps['max_complexity'] = current_time

        if self.max_smoothness is not None:
            np.save(os.path.join(self.evorun_dir, 'max_smoothness.npy'), self.max_smoothness)
            self.timestamps['max_smoothness'] = current_time

    def load_model(self):
        self._load_if_newer('pca', 'pca_model.pkl', self._load_pca)
        self._load_if_newer('pca_autoencoder', 'pca_autoencoder', self._load_pca_autoencoder)
        self._load_if_newer('scaler', 'scaler.pkl', self._load_scaler)
        self._load_if_newer('umap', 'umap_model', self._load_umap)
        self._load_if_newer('max_reconstruction_error', 'max_reconstruction_error.npy', self._load_max_reconstruction_error)
        self._load_if_newer('max_complexity', 'max_complexity.npy', self._load_max_complexity)
        self._load_if_newer('max_smoothness', 'max_smoothness.npy', self._load_max_smoothness)

    def _load_if_newer(self, key, filename, load_func):
        file_path = os.path.join(self.evorun_dir, filename)
        if os.path.exists(file_path):
            file_timestamp = os.path.getmtime(file_path)
            if key not in self.timestamps or file_timestamp > self.timestamps[key]:
                load_func(file_path)
                self.timestamps[key] = file_timestamp

    def _load_pca(self, path):
        self.pca = pkl.load(open(path, 'rb'))

    def _load_pca_autoencoder(self, path):
        self.pca_autoencoder = tf.keras.models.load_model(path)

    def _load_scaler(self, path):
        self.scaler = pkl.load(open(path, 'rb'))

    def _load_umap(self, path):
        self.umap = load_ParametricUMAP(path)

    def _load_max_reconstruction_error(self, path):
        self.max_reconstruction_error = np.load(path)

    def _load_max_complexity(self, path):
        self.max_complexity = np.load(path)

    def _load_max_smoothness(self, path):
        self.max_smoothness = np.load(path)

model_managers = {}

def get_model_manager(evorun_dir):
    if evorun_dir not in model_managers:
        model_managers[evorun_dir] = ModelManager(evorun_dir)
    return model_managers[evorun_dir]

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

def create_pca_autoencoder(input_dim, latent_dim, random_state):
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
    
    autoencoder = tf.keras.Model(inputs=encoder.input, outputs=decoder(encoder.output))
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def calculate_reconstruction_loss(model, features):
    if isinstance(model, ParametricUMAP):
        reconstructed = model.decoder(model.transform(features)).numpy()
    elif isinstance(model, PCA):
        if hasattr(model, 'autoencoder') and model.autoencoder is not None:
            reconstructed = model.autoencoder.predict(features)
        else:
            reconstructed = model.inverse_transform(model.transform(features))
    else:  # Autoencoder
        reconstructed = model.predict(features)
    return np.mean(np.square(features - reconstructed), axis=1)

def calculate_complexity(feature_vector):
    hist, _ = np.histogram(feature_vector, bins='auto', density=True)
    return entropy(hist + 1e-10)

def calculate_smoothness(feature_vector):
    total_variation = np.sum(np.abs(np.diff(feature_vector)))
    return 1 / (total_variation + 1e-10)

def calculate_surprise_score(reconstruction_loss, max_reconstruction_error, feature_vector, max_complexity, max_smoothness, alpha=10, beta=0.5, gamma=0.5):
    normalized_loss = reconstruction_loss / max_reconstruction_error
    complexity = calculate_complexity(feature_vector)
    smoothness = calculate_smoothness(feature_vector)
    
    normalized_complexity = complexity / max_complexity
    normalized_smoothness = smoothness / max_smoothness
    
    adjusted_loss = normalized_loss - beta * normalized_complexity + gamma * normalized_smoothness
    
    return 1 / (1 + np.exp(-alpha * (adjusted_loss - 0.5)))

def get_pca_projection(features, n_components=2, should_fit=True, evorun_dir='', calculate_surprise=False, components_list=[], use_autoencoder=False):
    model_manager = get_model_manager(evorun_dir)
    
    if should_fit:
        print('Fitting PCA model...')
        if len(components_list) > 0:
            model_manager.pca = PCA()
        else:
            model_manager.pca = PCA(n_components=n_components)
        model_manager.pca.fit(features)
        
        model_manager.scaler = MinMaxScaler(feature_range=(0, 1))
        transformed = model_manager.pca.transform(features)
        model_manager.scaler.fit(transformed)
        
        # TODO fine-tune after initial training, instead of training a new one from scratc on each retrainging pahes?
        if use_autoencoder:
            print('Training PCA autoencoder...')
            model_manager.pca_autoencoder = create_pca_autoencoder(features.shape[1], n_components, random_state=42)
            model_manager.pca_autoencoder.fit(features, features, epochs=100, batch_size=64, verbose=0)
        
        if calculate_surprise:
            if use_autoencoder:
                all_reconstruction_losses = calculate_reconstruction_loss(model_manager.pca_autoencoder, features)
            else:
                all_reconstruction_losses = calculate_reconstruction_loss(model_manager.pca, features)
            model_manager.max_reconstruction_error = np.max(all_reconstruction_losses)
            model_manager.max_complexity = np.max([calculate_complexity(f) for f in features])
            model_manager.max_smoothness = np.max([calculate_smoothness(f) for f in features])
        
        model_manager.save_model()
    else:
        model_manager.load_model()

    transformed = model_manager.pca.transform(features)
    scaled = model_manager.scaler.transform(transformed)

    if len(components_list) > 0:
        scaled = scaled[:, components_list]

    if calculate_surprise and model_manager.max_reconstruction_error is not None:
        if use_autoencoder and model_manager.pca_autoencoder is not None:
            reconstruction_losses = calculate_reconstruction_loss(model_manager.pca_autoencoder, features)
        else:
            reconstruction_losses = calculate_reconstruction_loss(model_manager.pca, features)
        surprise_scores = np.array([
            calculate_surprise_score(loss, model_manager.max_reconstruction_error, feature, model_manager.max_complexity, model_manager.max_smoothness)
            for loss, feature in zip(reconstruction_losses, features)
        ])
        return scaled, surprise_scores
    else:
        return scaled, None

def get_umap_projection(features, n_components=2, should_fit=True, evorun_dir='', calculate_surprise=False, random_state=42, n_neighbors=15, min_dist=0.1):
    model_manager = get_model_manager(evorun_dir)
    
    if should_fit:
        input_dim = features.shape[1]
        if calculate_surprise:
            encoder, decoder = create_autoencoder(input_dim, n_components, random_state)
        else:
            encoder, decoder = None, None
        
        model_manager.umap = ParametricUMAP(
            n_components=n_components,
            encoder=encoder,
            decoder=decoder,
            autoencoder_loss=calculate_surprise,
            loss_report_frequency=1,
            n_epochs=100,
            batch_size=64,
            random_state=random_state,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            init='random'
        )
        
        print('Fitting UMAP model...')
        model_manager.umap.fit(features)
        
        if calculate_surprise:
            all_reconstruction_losses = calculate_reconstruction_loss(model_manager.umap, features)
            model_manager.max_reconstruction_error = np.max(all_reconstruction_losses)
            model_manager.max_complexity = np.max([calculate_complexity(f) for f in features])
            model_manager.max_smoothness = np.max([calculate_smoothness(f) for f in features])
        
        model_manager.save_model()
    else:
        model_manager.load_model()

    transformed = model_manager.umap.transform(features)

    if calculate_surprise:
        reconstruction_losses = calculate_reconstruction_loss(model_manager.umap, features)
        surprise_scores = np.array([
            calculate_surprise_score(loss, model_manager.max_reconstruction_error, feature, model_manager.max_complexity, model_manager.max_smoothness)
            for loss, feature in zip(reconstruction_losses, features)
        ])
        return transformed, surprise_scores
    else:
        return transformed, None

def set_global_random_state(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)