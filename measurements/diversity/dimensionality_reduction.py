import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
import os
import pickle as pkl
from scipy.stats import entropy
import time
import warnings

class ModelManager:
    def __init__(self, evorun_dir):
        self.evorun_dir = evorun_dir
        self.pca = None
        self.pca_autoencoder = None
        self.pca_encoder = None
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
            pca_ae_path = os.path.join(self.evorun_dir, 'pca_autoencoder.keras')
            self.pca_autoencoder.save(pca_ae_path)
            self.timestamps['pca_autoencoder'] = current_time

        if self.pca_encoder is not None:
            pca_encoder_path = os.path.join(self.evorun_dir, 'pca_encoder.keras')
            self.pca_encoder.save(pca_encoder_path)
            self.timestamps['pca_encoder'] = current_time

        if self.scaler is not None:
            scaler_path = os.path.join(self.evorun_dir, 'scaler.pkl')
            pkl.dump(self.scaler, open(scaler_path, 'wb'))
            self.timestamps['scaler'] = current_time

        if self.umap is not None:
            umap_path = os.path.join(self.evorun_dir, 'umap_model')
            os.makedirs(umap_path, exist_ok=True)  # Ensure the directory itself is created
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
        self._load_if_newer('pca_autoencoder', 'pca_autoencoder.keras', self._load_pca_autoencoder)
        self._load_if_newer('pca_encoder', 'pca_encoder.keras', self._load_pca_encoder)
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

    def _load_pca_encoder(self, path):
        self.pca_encoder = tf.keras.models.load_model(path)

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
    
    inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    encoded = tf.keras.layers.Dense(latent_dim, activation='linear', name='encoder_output')(x)
    
    x = tf.keras.layers.Dense(32, activation='relu')(encoded)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    
    autoencoder = tf.keras.Model(inputs=inputs, outputs=outputs)
    encoder = tf.keras.Model(inputs=inputs, outputs=encoded)
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder

def calculate_reconstruction_loss(model, features):
    if isinstance(model, ParametricUMAP):
        if hasattr(model, 'encoder') and hasattr(model, 'decoder') and model.encoder is not None and model.decoder is not None:
            reconstructed = model.decoder(model.transform(features)).numpy()
        else:
            # If ParametricUMAP doesn't have encoder and decoder, use transform and inverse_transform
            transformed = model.transform(features)
            reconstructed = model.inverse_transform(transformed)
    elif isinstance(model, PCA):
        if hasattr(model, 'autoencoder') and model.autoencoder is not None:
            reconstructed = model.autoencoder.predict(features)
        else:
            reconstructed = model.inverse_transform(model.transform(features))
    else:  # Autoencoder
        if hasattr(model, 'predict'):
            reconstructed = model.predict(features)
        elif hasattr(model, 'encoder') and hasattr(model, 'decoder'):
            encoded = model.encoder(features)
            reconstructed = model.decoder(encoded).numpy()
        else:
            raise ValueError("Unsupported model type for reconstruction loss calculation")
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

def get_pca_projection(features, n_components=2, should_fit=True, evorun_dir='', calculate_surprise=False, components_list=[], use_autoencoder=False, dynamic_components=False):
    model_manager = get_model_manager(evorun_dir)
    
    # Type checking and conversion
    if isinstance(features, list):
        features = np.array(features, dtype=np.float32)
    elif not isinstance(features, np.ndarray):
        raise ValueError(f"Unrecognized data type for features: {type(features)}")
    
    print(f"Features shape: {features.shape}")
    print(f"Features dtype: {features.dtype}")
    
    # Initialize variables for feature and PCA analysis
    feature_contribution = None
    feature_indices = None
    selected_pca_components = None
    
    if should_fit:
        print('Fitting PCA model...')
        # First fit PCA with all components
        model_manager.pca = PCA()
        model_manager.pca.fit(features)
        
        # Analyze feature contributions to PCA components
        # Get the absolute values of component loadings
        loadings = np.abs(model_manager.pca.components_)
        
        # Calculate the contribution of each input feature
        feature_contribution = np.mean(loadings, axis=0)
        
        # Select features based on their contributions
        # Here we use a threshold based on the mean contribution
        threshold = np.mean(feature_contribution)
        feature_indices = np.where(feature_contribution > threshold)[0]
        
        # If components_list is empty, select the most informative PCA components
        if len(components_list) == 0:
            # Calculate the cumulative explained variance ratio
            cumsum = np.cumsum(model_manager.pca.explained_variance_ratio_)
            # Select components that explain up to 95% of variance
            n_components_95 = np.argmax(cumsum >= 0.95) + 1
            # Take either n_components or the number needed for 95% variance, whichever is smaller
            n_select = min(n_components, n_components_95)
            components_list = list(range(n_select))
        
        selected_pca_components = components_list
        
        # Create new PCA model with selected components
        if len(components_list) > 0 and dynamic_components:
            model_manager.pca = PCA(n_components=len(components_list))
            # Use only selected features
            features_selected = features[:, feature_indices]
            model_manager.pca.fit(features_selected)
        else:
            model_manager.pca = PCA(n_components=n_components)
            model_manager.pca.fit(features)
            feature_indices = None  # Reset feature_indices if not using dynamic components
        
        model_manager.scaler = MinMaxScaler(feature_range=(0, 1))
        # Check if feature_indices exists and is not None before trying to use it
        features_to_transform = features[:, feature_indices] if feature_indices is not None and len(feature_indices) > 0 else features
        transformed = model_manager.pca.transform(features_to_transform)
        model_manager.scaler.fit(transformed)
        del transformed
        
        # Handle surprise calculation (existing code)
        if calculate_surprise and use_autoencoder:
            if model_manager.pca_autoencoder is None:
                print('Initializing PCA autoencoder for surprise calculation...')
                model_manager.pca_autoencoder, model_manager.pca_encoder = create_pca_autoencoder(
                    len(feature_indices) if len(feature_indices) > 0 else features.shape[1],
                    n_components, 
                    random_state=42
                )
            print('Fine-tuning PCA autoencoder...')
            features_for_ae = features[:, feature_indices] if len(feature_indices) > 0 else features
            model_manager.pca_autoencoder.fit(features_for_ae, features_for_ae, epochs=10, batch_size=64, verbose=0)
        
        if calculate_surprise:
            features_for_surprise = features[:, feature_indices] if len(feature_indices) > 0 else features
            if use_autoencoder and model_manager.pca_autoencoder is not None:
                all_reconstruction_losses = calculate_reconstruction_loss(model_manager.pca_autoencoder, features_for_surprise)
            else:
                all_reconstruction_losses = calculate_reconstruction_loss(model_manager.pca, features_for_surprise)
            model_manager.max_reconstruction_error = np.max(all_reconstruction_losses)
            model_manager.max_complexity = np.max([calculate_complexity(f) for f in features_for_surprise])
            model_manager.max_smoothness = np.max([calculate_smoothness(f) for f in features_for_surprise])
            del all_reconstruction_losses
        
        model_manager.save_model()
    else:
        model_manager.load_model()

    # Use selected features for transformation
    features_to_transform = features[:, feature_indices] if feature_indices is not None else features
    transformed = model_manager.pca.transform(features_to_transform)
    scaled = model_manager.scaler.transform(transformed)
    del transformed

    if len(components_list) > 0:
        scaled = scaled[:, components_list]

    # Calculate surprise scores if requested
    surprise_scores = None
    if calculate_surprise and model_manager.max_reconstruction_error is not None:
        features_for_surprise = features[:, feature_indices] if feature_indices is not None else features
        if use_autoencoder and model_manager.pca_autoencoder is not None:
            reconstruction_losses = calculate_reconstruction_loss(model_manager.pca_autoencoder, features_for_surprise)
        else:
            reconstruction_losses = calculate_reconstruction_loss(model_manager.pca, features_for_surprise)
        
        surprise_scores = np.array([
            calculate_surprise_score(
                loss, 
                model_manager.max_reconstruction_error or 1, 
                feature, 
                model_manager.max_complexity or 1, 
                model_manager.max_smoothness or 1
            )
            for loss, feature in zip(reconstruction_losses, features_for_surprise)
        ])
        del reconstruction_losses

    return (scaled, 
            surprise_scores, 
            feature_contribution, 
            feature_indices, 
            selected_pca_components)

def get_autoencoder_projection(features, n_components=2, should_fit=True, evorun_dir='', calculate_surprise=False, random_state=42):
    model_manager = get_model_manager(evorun_dir)
    
    # Ensure features is a 2D numpy array
    features = np.array(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    print(f"Features shape: {features.shape}")
    
    if should_fit:
        print('Fitting autoencoder...')
        if model_manager.pca_autoencoder is None:
            print('Initializing new autoencoder...')
            model_manager.pca_autoencoder, model_manager.pca_encoder = create_pca_autoencoder(features.shape[1], n_components, random_state=random_state)
        
        model_manager.pca_autoencoder.fit(features, features, epochs=10, batch_size=64, verbose=1)
        
        # Update scaler for consistent output range
        encoded_features = model_manager.pca_encoder.predict(features)
        model_manager.scaler = MinMaxScaler(feature_range=(0, 1))
        model_manager.scaler.fit(encoded_features)
        
        if calculate_surprise:
            all_reconstruction_losses = calculate_reconstruction_loss(model_manager.pca_autoencoder, features)
            model_manager.max_reconstruction_error = np.max(all_reconstruction_losses)
            model_manager.max_complexity = np.max([calculate_complexity(f) for f in features])
            model_manager.max_smoothness = np.max([calculate_smoothness(f) for f in features])
        
        model_manager.save_model()
    else:
        model_manager.load_model()
    
    if model_manager.pca_encoder is None:
        raise ValueError("Encoder not initialized. Please run with should_fit=True first.")

    print(f"Encoder input shape: {model_manager.pca_encoder.input_shape}")
    print(f"Features shape before encoding: {features.shape}")
    
    # Ensure features match the expected input shape
    if features.shape[1:] != model_manager.pca_encoder.input_shape[1:]:
        raise ValueError(f"Feature shape {features.shape[1:]} does not match encoder input shape {model_manager.pca_encoder.input_shape[1:]}")
    
    # Use autoencoder for projection
    try:
        encoded_features = model_manager.pca_encoder.predict(features)
        print(f"Encoded features shape: {encoded_features.shape}")
    except Exception as e:
        print(f"Error during encoding: {str(e)}")
        print("Encoder summary:")
        model_manager.pca_encoder.summary()
        raise

    scaled = model_manager.scaler.transform(encoded_features)

    if calculate_surprise and model_manager.max_reconstruction_error is not None:
        reconstruction_losses = calculate_reconstruction_loss(model_manager.pca_autoencoder, features)
        surprise_scores = np.array([
            calculate_surprise_score(
                loss, 
                model_manager.max_reconstruction_error or 1, 
                feature, 
                model_manager.max_complexity or 1, 
                model_manager.max_smoothness or 1
            )
            for loss, feature in zip(reconstruction_losses, features)
        ])
        return scaled, surprise_scores
    else:
        return scaled, None

def get_umap_projection(features, n_components=2, should_fit=True, evorun_dir='', calculate_surprise=False, 
                        random_state=42, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    model_manager = get_model_manager(evorun_dir)
    
    features = np.array(features)
    n_samples = features.shape[0]

    # Determine appropriate n_neighbors and batch size
    original_n_neighbors = n_neighbors
    if n_samples <= n_neighbors:
        n_neighbors = max(2, n_samples - 1)
        warnings.warn(f"n_neighbors ({original_n_neighbors}) is greater than or equal to n_samples ({n_samples}). "
                      f"Reducing n_neighbors to {n_neighbors} for this projection.")
    
    if should_fit:
        input_dim = features.shape[1]

        if calculate_surprise:
            encoder, decoder = create_autoencoder(input_dim, n_components, random_state)
        else:
            encoder, decoder = None, None
        
        # Adjust batch size and epochs for small sample sizes
        batch_size = min(64, max(1, n_samples))  # Ensure batch_size <= n_samples
        n_epochs = 200 if n_samples < 10 else 100  # More epochs for small samples
        
        model_manager.umap = ParametricUMAP(
            n_components=n_components,
            encoder=encoder,
            decoder=decoder,
            autoencoder_loss=calculate_surprise,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
        )
        
        print(f'Fitting UMAP model with batch_size={batch_size}, n_epochs={n_epochs}...')
        model_manager.umap.fit(features)
        
        if calculate_surprise:
            all_reconstruction_losses = calculate_reconstruction_loss(model_manager.umap, features)
            model_manager.max_reconstruction_error = np.max(all_reconstruction_losses)
            model_manager.max_complexity = np.max([calculate_complexity(f) for f in features])
            model_manager.max_smoothness = np.max([calculate_smoothness(f) for f in features])
        
        model_manager.save_model()
    else:
        model_manager.load_model()

    # Rest of the function remains the same...
    if n_samples <= model_manager.umap.n_neighbors:
        umap_params = {
            'n_components': model_manager.umap.n_components,
            'n_neighbors': n_neighbors,
            'min_dist': model_manager.umap.min_dist,
            'metric': model_manager.umap.metric,
        }
        
        if hasattr(model_manager.umap, 'encoder') and model_manager.umap.encoder is not None:
            umap_params['encoder'] = model_manager.umap.encoder
        if hasattr(model_manager.umap, 'decoder') and model_manager.umap.decoder is not None:
            umap_params['decoder'] = model_manager.umap.decoder
        
        for attr in ['n_epochs', 'batch_size', 'autoencoder_loss']:
            if hasattr(model_manager.umap, attr):
                umap_params[attr] = getattr(model_manager.umap, attr)
        
        temp_umap = ParametricUMAP(**umap_params)
        
        if hasattr(model_manager.umap, 'embedding_'):
            temp_umap.embedding_ = model_manager.umap.embedding_
        
        transformed = temp_umap.transform(features)
    else:
        transformed = model_manager.umap.transform(features)

    if calculate_surprise:
        reconstruction_losses = calculate_reconstruction_loss(model_manager.umap, features)
        surprise_scores = np.array([
            calculate_surprise_score(
                loss, 
                model_manager.max_reconstruction_error or 1, 
                feature, 
                model_manager.max_complexity or 1, 
                model_manager.max_smoothness or 1
            )
            for loss, feature in zip(reconstruction_losses, features)
        ])
        return transformed, surprise_scores
    else:
        return transformed, None

def set_global_random_state(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

import tensorflow as tf
def clear_tf_session():
    tf.keras.backend.clear_session()

def projection_with_cleanup(projection_func, *args, **kwargs):
    try:
        result = projection_func(*args, **kwargs)
        return result
    finally:
        # Clear any leftover tensors
        tf.keras.backend.clear_session()
        
        # Force garbage collection
        import gc
        gc.collect()