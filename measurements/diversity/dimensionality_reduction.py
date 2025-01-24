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

def create_vae(input_dim, latent_dim, random_state):
    tf.random.set_seed(random_state)
    
    # Encoder
    encoder_inputs = tf.keras.Input(shape=(input_dim,))
    x = tf.keras.layers.Dense(64, activation='relu')(encoder_inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    
    # VAE specific: create mean and log variance layers
    z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)
    
    # Sampling layer
    class Sampling(tf.keras.layers.Layer):
        def call(self, inputs):
            z_mean, z_log_var = inputs
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = Sampling()([z_mean, z_log_var])
    
    # Build encoder model
    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    
    # Decoder
    decoder_inputs = tf.keras.Input(shape=(latent_dim,))
    x = tf.keras.layers.Dense(32, activation='relu')(decoder_inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    decoder_outputs = tf.keras.layers.Dense(input_dim, activation='linear')(x)
    decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name='decoder')
    
    # VAE model
    class VAE(tf.keras.Model):
        def __init__(self, encoder, decoder, **kwargs):
            super().__init__(**kwargs)
            self.encoder = encoder
            self.decoder = decoder
            self.total_loss_tracker = tf.keras.metrics.Mean(name='total_loss')
            self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name='reconstruction_loss')
            self.kl_loss_tracker = tf.keras.metrics.Mean(name='kl_loss')
            
        @property
        def metrics(self):
            return [
                self.total_loss_tracker,
                self.reconstruction_loss_tracker,
                self.kl_loss_tracker,
            ]
        
        def train_step(self, data):
            with tf.GradientTape() as tape:
                z_mean, z_log_var, z = self.encoder(data)
                reconstruction = self.decoder(z)
                reconstruction_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.keras.losses.mse(data, reconstruction), axis=1
                    )
                )
                kl_loss = -0.5 * tf.reduce_mean(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
                )
                total_loss = reconstruction_loss + kl_loss
                
            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
            
            self.total_loss_tracker.update_state(total_loss)
            self.reconstruction_loss_tracker.update_state(reconstruction_loss)
            self.kl_loss_tracker.update_state(kl_loss)
            
            return {
                "loss": self.total_loss_tracker.result(),
                "reconstruction_loss": self.reconstruction_loss_tracker.result(),
                "kl_loss": self.kl_loss_tracker.result(),
            }
    
    vae = VAE(encoder, decoder)
    vae.compile(optimizer='adam')
    
    return vae, encoder

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

def get_pca_projection(features, n_components=2, should_fit=True, evorun_dir='', calculate_surprise=False, 
                      components_list=[], use_autoencoder=False, dynamic_components=False,
                      selection_strategy='improved', selection_params=None):
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
    selected_pca_components = components_list
    component_contribution = None
    
    if should_fit:
        print('Fitting PCA model...')
        
        if dynamic_components:
            # First fit PCA with all components to analyze contributions
            model_manager.pca = PCA()
            model_manager.pca.fit(features)
            
            # Analyze feature contributions
            loadings = np.abs(model_manager.pca.components_)
            feature_contribution = np.mean(loadings, axis=0)
            
            # Set up selection parameters
            if selection_strategy == 'original': # "original" seems to eventually result in zero length feature indices
                threshold = np.mean(feature_contribution)
                if selection_params and 'threshold_multiplier' in selection_params:
                    threshold *= selection_params['threshold_multiplier']
                feature_indices = np.where(feature_contribution > threshold)[0]
                
            elif selection_strategy == 'improved':
                min_features_pct = selection_params.get('min_features_pct', 0.1) if selection_params else 0.1
                variance_threshold = selection_params.get('variance_threshold', 0.95) if selection_params else 0.95
                use_sliding_window = selection_params.get('use_sliding_window', True) if selection_params else False
                
                if len(feature_contribution) > 0:
                    sorted_indices = np.argsort(feature_contribution)[::-1]
                    # Reorder feature_contribution to match feature_indices order
                    feature_contribution = feature_contribution[sorted_indices]
                    n_features = len(feature_contribution)
                    min_features = max(2, int(n_features * min_features_pct))
                    
                    cumsum_contrib = np.cumsum(feature_contribution[sorted_indices])
                    cumsum_contrib /= cumsum_contrib[-1]
                    n_features_variance = np.argmax(cumsum_contrib >= variance_threshold) + 1
                    n_features_keep = max(min_features, n_features_variance)
                    
                    current_features = sorted_indices[:n_features_keep]
                    
                    if use_sliding_window and hasattr(model_manager, 'previous_feature_indices'):
                        previous_features = model_manager.previous_feature_indices
                        stable_features = np.union1d(
                            current_features,
                            np.intersect1d(previous_features, sorted_indices[:len(previous_features)])
                        )
                        feature_indices = stable_features
                    else:
                        feature_indices = current_features
                    
                    if use_sliding_window:
                        model_manager.previous_feature_indices = feature_indices
                    
                    if len(feature_indices) < 2:
                        print("Warning: Too few features selected, using top 2 features")
                        feature_indices = sorted_indices[:2]
                        
                # Log selection details
                print(f"Strategy: {selection_strategy}")
                print(f"Features selected: {len(feature_indices)}")
                if len(feature_indices) > 0:
                    print(f"Avg contribution: {np.mean(feature_contribution[feature_indices]):.4f}")
                    print(f"Min/Max contrib: {np.min(feature_contribution[feature_indices]):.4f} / {np.max(feature_contribution[feature_indices]):.4f}")
            else:
                raise ValueError(f"Unknown selection strategy: {selection_strategy}")
        
        # Create new PCA model
        features_to_use = features[:, feature_indices] if dynamic_components and feature_indices is not None else features
        
        if len(components_list) == 0 and dynamic_components:
            cumsum = np.cumsum(model_manager.pca.explained_variance_ratio_)
            n_components_95 = np.argmax(cumsum >= 0.95) + 1
            n_select = min(n_components, n_components_95)
            components_list = list(range(n_select))
        
        selected_pca_components = components_list

        if len(components_list) > 0 and dynamic_components:
            model_manager.pca = PCA(n_components=len(components_list))
        else:
            model_manager.pca = PCA(n_components=n_components)
        
        model_manager.pca.fit(features_to_use)
        
        # Setup scaler
        model_manager.scaler = MinMaxScaler(feature_range=(0, 1))
        transformed = model_manager.pca.transform(features_to_use)
        model_manager.scaler.fit(transformed)
        del transformed
        
        # Handle surprise calculation
        if calculate_surprise:
            if use_autoencoder:
                if model_manager.pca_autoencoder is None:
                    print('Initializing PCA autoencoder...')
                    model_manager.pca_autoencoder, model_manager.pca_encoder = create_pca_autoencoder(
                        features_to_use.shape[1],
                        n_components,
                        random_state=42
                    )
                print('Fine-tuning PCA autoencoder...')
                model_manager.pca_autoencoder.fit(features_to_use, features_to_use, epochs=10, batch_size=64, verbose=0)
            
            if use_autoencoder and model_manager.pca_autoencoder is not None:
                all_reconstruction_losses = calculate_reconstruction_loss(model_manager.pca_autoencoder, features_to_use)
            else:
                all_reconstruction_losses = calculate_reconstruction_loss(model_manager.pca, features_to_use)
            
            model_manager.max_reconstruction_error = np.max(all_reconstruction_losses)
            model_manager.max_complexity = np.max([calculate_complexity(f) for f in features_to_use])
            model_manager.max_smoothness = np.max([calculate_smoothness(f) for f in features_to_use])
            del all_reconstruction_losses
        
        # Calculate component contributions after fitting PCA
            # - explained_variance_ratio_:
            # Shows how much of the total variance in the data is explained by each principal component
            # Values are between 0 and 1 (or 0-100%)
            # Sum of all ratios equals 1 (100%)
            # Higher values indicate more important components
            # Example: [0.5, 0.3, 0.15, 0.05] means the first component explains 50% of variance, second 30%, etc.
            # - cumulative_variance_ratio:
            # Running sum of explained_variance_ratio_
            # Shows total variance explained up to each component
            # Helps determine how many components to keep
            # Always increases, reaching 1.0 (100%) at the end
            # Example: [0.5, 0.8, 0.95, 1.0] means first two components together explain 80% of variance
            # - singular_values_:
            # Square roots of eigenvalues from the SVD (Singular Value Decomposition)
            # Represent the "strength" or "scale" of each principal component
            # Larger values indicate more important components
            # Not normalized (unlike variance ratios)
            # Related to explained variance: (singular_value²) / (n_samples - 1) = explained_variance
            # - Key differences:
            # explained_variance_ratio_ is normalized (0-1) and shows relative importance
            # cumulative_variance_ratio shows accumulated explanation power
            # singular_values_ shows absolute scale/magnitude of components in the original data space
        component_contribution = {
            'explained_variance_ratio': model_manager.pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(model_manager.pca.explained_variance_ratio_),
            'singular_values': model_manager.pca.singular_values_
        }

        model_manager.save_model()
    else:
        model_manager.load_model()
        # Get component contributions from loaded model
        component_contribution = {
            'explained_variance_ratio': model_manager.pca.explained_variance_ratio_,
            'cumulative_variance_ratio': np.cumsum(model_manager.pca.explained_variance_ratio_),
            'singular_values': model_manager.pca.singular_values_
        }

    # Transform features using the fitted model
    features_to_transform = features[:, feature_indices] if dynamic_components and feature_indices is not None else features
    transformed = model_manager.pca.transform(features_to_transform)
    scaled = model_manager.scaler.transform(transformed)
    del transformed

    # Calculate surprise scores if requested
    surprise_scores = None
    if calculate_surprise and model_manager.max_reconstruction_error is not None:
        features_for_surprise = features_to_transform
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

    return (scaled, surprise_scores, feature_contribution, feature_indices, selected_pca_components, component_contribution)

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

def get_vae_projection(features, n_components=2, should_fit=True, evorun_dir='', calculate_surprise=False, random_state=42):
    model_manager = get_model_manager(evorun_dir)
    
    features = np.array(features, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    if should_fit:
        print('Fitting VAE...')
        vae, encoder = create_vae(features.shape[1], n_components, random_state)
        model_manager.pca_autoencoder = vae  # Reuse existing attribute
        model_manager.pca_encoder = encoder  # Reuse existing attribute
        
        model_manager.pca_autoencoder.fit(features, epochs=50, batch_size=64, verbose=1)
        
        # Update scaler for consistent output range
        _, _, encoded_features = model_manager.pca_encoder(features)
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
    
    # Get projection
    _, _, encoded_features = model_manager.pca_encoder(features)
    scaled = model_manager.scaler.transform(encoded_features)
    
    if calculate_surprise and model_manager.max_reconstruction_error is not None:
        reconstruction_losses = calculate_reconstruction_loss(model_manager.pca_autoencoder, features)
        surprise_scores = np.array([
            calculate_surprise_score(
                loss, 
                model_manager.max_reconstruction_error,
                feature, 
                model_manager.max_complexity,
                model_manager.max_smoothness
            )
            for loss, feature in zip(reconstruction_losses, features)
        ])
        return scaled, surprise_scores
    
    return scaled, None

def get_umap_projection(features, n_components=2, should_fit=True, evorun_dir='', calculate_surprise=False, 
                        random_state=42, n_neighbors=15, min_dist=0.1, metric='euclidean'):
    model_manager = get_model_manager(evorun_dir)
    
    features = np.array(features)
    n_samples = features.shape[0]

    # Special handling for very small datasets
    if n_samples < 4:  # UMAP needs at least 4 samples to work reliably
        print(f"Warning: Dataset too small for UMAP (n_samples={n_samples}). Falling back to PCA.")
        # Always use should_fit=True for PCA fallback when in fitting mode
        use_fit = True if should_fit else False
        pca_result = get_pca_projection(
            features, 
            n_components=n_components, 
            should_fit=use_fit, 
            evorun_dir=evorun_dir, 
            calculate_surprise=calculate_surprise
        )
        # Extract just the projection and surprise scores from PCA result
        if calculate_surprise:
            return pca_result[0], pca_result[1]
        return pca_result[0], None

    if should_fit:
        # Only proceed with UMAP fitting if we have enough samples
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