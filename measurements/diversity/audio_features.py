import librosa
import numpy as np
from frechet_audio_distance import FrechetAudioDistance
from essentia.standard import TensorflowPredictVGGish, TensorflowPredictMAEST, TensorflowPredictEffnetDiscogs, TensorflowPredictMusiCNN
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import openl3
from transformers import AutoProcessor, ASTModel, AutoModel, AutoFeatureExtractor # ASTFeatureExtractor
import torch
import pyACA
import sklearn

# idea from: https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f

def get_feature_means_stdv_firstorderdifference_concatenated(features):
    num_features = len(features)
    # Transpose the array so that the shape is (num_features, num_frames)
    features = features.T

    # Calculate the mean and standard deviation of each MFCC feature
    feature_means = np.mean(features, axis=0)
    feature_stdv = np.std(features, axis=0)

    # Get the average difference of the features (first order difference)
    average_difference_features = np.zeros((num_features,))
    for i in range(0, len(features) - 2, 2):
        average_difference_features += features[i] - features[i+1]
    average_difference_features /= (len(features) // 2)   
    average_difference_features = np.array(average_difference_features)

    # Concatenate the MFCC features for each frame into a single array
    features_concatenated = np.hstack((feature_means, feature_stdv, average_difference_features))

    return features_concatenated

# sample_rate should be 16000 for VGGish
def get_vggish_embeddings(audio_data, sample_rate, ckpt_dir, use_pca=False, use_activation=False):

    # TODO replace with?: https://www.kaggle.com/models/google/vggish

    if sample_rate != 16000:
        # Resample the audio data to 16 kHz
        audio_data[0] = librosa.resample(y=audio_data[0], orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    frechet = FrechetAudioDistance(
        ckpt_dir=ckpt_dir,
        model_name="vggish",
        # submodel_name="630k-audioset", # for CLAP only
        sample_rate=sample_rate,
        use_pca=use_pca, # for VGGish only
        use_activation=use_activation, # for VGGish only
        verbose=True,
        audio_load_worker=8,
        # enable_fusion=False, # for CLAP only
    )
    embeddings = frechet.get_embeddings(audio_data, sample_rate)
    return embeddings

def get_vggish_embeddings_essentia(audio_data, sample_rate, models_path):
    if sample_rate != 16000:
        # Resample the audio data to 16 kHz
        audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    # https://essentia.upf.edu/models.html#audioset-vggish
    # https://doi.org/10.1109/ICASSP40776.2020.9054688
    VGGISH_MODEL_PATH=f"{models_path}/audioset-vggish-3.pb"
    print('VGGish model path:', VGGISH_MODEL_PATH)
    model = TensorflowPredictVGGish(graphFilename=VGGISH_MODEL_PATH, output="model/vggish/embeddings")
    embeddings = model(audio_data)
    return embeddings

# sample_rate should be 8000, 16000 or 32000 for PANN
def get_pann_embeddings(audio_data, sample_rate, ckpt_dir):
    if sample_rate > 32000:
        # Resample the audio data to 32 kHz
        audio_data[0] = librosa.resample(y=audio_data[0], orig_sr=sample_rate, target_sr=32000)
        sample_rate = 32000
    elif sample_rate > 16000:
        # Resample the audio data to 16 kHz
        audio_data[0] = librosa.resample(y=audio_data[0], orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    elif sample_rate > 8000:
        # Resample the audio data to 8 kHz
        audio_data[0] = librosa.resample(y=audio_data[0], orig_sr=sample_rate, target_sr=8000)
        sample_rate = 8000
    frechet = FrechetAudioDistance(
        ckpt_dir=ckpt_dir,
        model_name="pann",
        sample_rate=sample_rate,
        verbose=True,
        audio_load_worker=8,
    )
    embeddings = frechet.get_embeddings(audio_data, sample_rate)
    return embeddings

# https://github.com/qiuqiangkong/panns_inference?tab=readme-ov-file
# [1] Kong, Qiuqiang, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. "PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition." arXiv preprint arXiv:1912.10211 (2019).
def get_pann_embeddings_panns_inference(audio_data, sample_rate, ckpt_dir):
    if sample_rate != 32000:
        # Resample the audio data to 32 kHz
        audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=32000)
        sample_rate = 32000
    at = AudioTagging(checkpoint_path=ckpt_dir) # model_type='PANNsCNN14Att'
    audio_data = audio_data[None, :]
    print('audio_data shape:', audio_data.shape)
    (clipwise_output, embedding) = at.inference(audio_data)
    return embedding

# sample_rate should be 48000 for CLAP
# submodel options (from https://github.com/gudgud96/frechet-audio-distance/blob/main/test/test_all.ipynb):
# 630k-audioset (for general audio less than 10-sec)
# 630k-audioset + fusion (for general audio with variable-length)
# 630k (for general audio less than 10-sec)
# 630k + fusion (for general audio with variable-length)
# music_audioset (for music)
# - (trained on music + Audioset + LAION-Audio-630k. The zeroshot ESC50 performance is 90.14%, the GTZAN performance is 71%.)
# music_speech (for music and speech)
# - trained on music + speech + LAION-Audio-630k. The zeroshot ESC50 performance is 89.25%, the GTZAN performance is 69%.
# - music_speech_audioset (for speech, music and general audio)
# trained on music + speech + Audioset + LAION-Audio-630k. The zeroshot ESC50 performance is 89.98%, the GTZAN performance is 51%.
def get_clap_embeddings(audio_data, sample_rate, ckpt_dir, submodel_name="630k-audioset", enable_fusion=False):
    print('submodel_name:', submodel_name)
    print('enable_fusion:', enable_fusion)
    if sample_rate != 48000:
        # Resample the audio data to 48 kHz
        audio_data[0] = librosa.resample(y=audio_data[0], orig_sr=sample_rate, target_sr=48000)
        sample_rate = 48000
    frechet = FrechetAudioDistance(
        ckpt_dir=ckpt_dir,
        model_name="clap",
        sample_rate=sample_rate,
        verbose=True,
        audio_load_worker=8,
        submodel_name=submodel_name,
        enable_fusion=enable_fusion
    )
    embeddings = frechet.get_embeddings(audio_data, sample_rate)
    print('CLAP embeddings shape:', embeddings.shape)
    return embeddings

# sample_rate should be 24000 or 48000 (then stereo) for EnCodec
def get_encodec_embeddings(audio_data, sample_rate, ckpt_dir):
    if sample_rate != 24000:
        # Resample the audio data to 24 kHz
        audio_data[0] = librosa.resample(y=audio_data[0], orig_sr=sample_rate, target_sr=24000)
        sample_rate = 24000
    frechet = FrechetAudioDistance(
        ckpt_dir=ckpt_dir,
        model_name="encodec",
        sample_rate=sample_rate,
        verbose=True,
        audio_load_worker=8,
    )
    embeddings = frechet.get_embeddings(audio_data, sample_rate)
    return embeddings


# @inproceedings{alonso2023Efficient,
#   title={Efficient Supervised Training of Audio Transformers for Music Representation Learning},
#   author={Pablo Alonso-Jim{\'e}nez and Xavier Serra and Dmitry Bogdanov},
#   booktitle={Proceedings of the International Society for Music Information Retrieval Conference},
#   year={2023},
# }

# MAEST 
# - https://essentia.upf.edu/models.html#maest
# - https://repositori.upf.edu/handle/10230/58023
maest_model = None
def get_maest_embeddings(audio_data, sample_rate, models_path):
    global maest_model
    if sample_rate != 16000:
        # Resample the audio data to 16 kHz
        audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    # if the input signal is shorter than 5 seconds, it will be zero-padded to 5 seconds
    if len(audio_data) < 5 * sample_rate:
        audio_data = librosa.util.fix_length(audio_data, size=(6 * sample_rate))
    # https://essentia.upf.edu/models.html#maest
    # https://repositori.upf.edu/handle/10230/58023
    if maest_model is None:
        MAEST_MODEL_PATH=f"{models_path}/discogs-maest-5s-pw-1.pb"
        print('MAEST model path:', MAEST_MODEL_PATH)
        maest_model = TensorflowPredictMAEST(graphFilename=MAEST_MODEL_PATH, output="StatefulPartitionedCall:7")
    embeddings = maest_model(audio_data)
    return embeddings


# Discogs-EffNet https://essentia.upf.edu/models.html#discogs-effnet
discogs_effnet_model = None
def get_discogs_effnet_embeddings(audio_data, sample_rate, models_path):
    global discogs_effnet_model
    if sample_rate != 16000:
        # Resample the audio data to 16 kHz
        audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    # https://essentia.upf.edu/models.html#discogs-effnet
    DISCOGS_EFFNET_MODEL_PATH=f"{models_path}/discogs-effnet-bs64-1.pb"
    print('Discogs-EffNet model path:', DISCOGS_EFFNET_MODEL_PATH)
    if discogs_effnet_model is None:
        discogs_effnet_model = TensorflowPredictEffnetDiscogs(graphFilename=DISCOGS_EFFNET_MODEL_PATH, output="PartitionedCall:1")
    embeddings = discogs_effnet_model(audio_data)
    return embeddings

# TODO MSD-MusiCNN https://essentia.upf.edu/models.html#msd-musicnn
msd_musicnn_model = None
def get_msd_musicnn_embeddings(audio_data, sample_rate, models_path):
    global msd_musicnn_model
    if sample_rate != 16000:
        # Resample the audio data to 16 kHz
        audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    # https://essentia.upf.edu/models.html#msd-musicnn
    MSD_MUSICNN_MODEL_PATH=f"{models_path}/msd-musicnn-1.pb"
    print('MSD-MusiCNN model path:', MSD_MUSICNN_MODEL_PATH)
    if msd_musicnn_model is None:
        msd_musicnn_model = TensorflowPredictMusiCNN(graphFilename=MSD_MUSICNN_MODEL_PATH, output="model/dense/BiasAdd")
    embeddings = msd_musicnn_model(audio_data)
    return embeddings


# Wav2Vec
wav2vec_model = None
wav2vec_feature_extractor = None
def get_wav2vec_embeddings(audio_data, sample_rate):
    global wav2vec_model, wav2vec_feature_extractor
    if sample_rate != 16000:
        # Resample the audio data to 16 kHz
        audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if wav2vec_model is None:
        wav2vec_model = AutoModel.from_pretrained('facebook/wav2vec2-base').to(device)
    if wav2vec_feature_extractor is None:
        wav2vec_feature_extractor = AutoFeatureExtractor.from_pretrained('facebook/wav2vec2-base')

    inputs = wav2vec_feature_extractor(
        audio_data, sampling_rate=wav2vec_feature_extractor.sampling_rate, return_tensors="pt",
        padding=True, return_attention_mask=True, truncation=True, max_length=16_000
    ).to(device)

    with torch.no_grad():
        embeddings = wav2vec_model(**inputs).last_hidden_state.mean(dim=1)

    return embeddings.numpy()


# ASTFeatureExtractor https://huggingface.co/docs/transformers/v4.40.1/en/model_doc/audio-spectrogram-transformer#transformers.ASTFeatureExtractor (https://huggingface.co/docs/transformers/model_doc/audio-spectrogram-transformer)
# https://arxiv.org/abs/2104.01778

# [1] Look, Listen and Learn More: Design Choices for Deep Audio Embeddings
# Aurora Cramer, Ho-Hsiang Wu, Justin Salamon, and Juan Pablo Bello.
# IEEE Int. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pages 3852–3856, Brighton, UK, May 2019.

# [2] Look, Listen and Learn
# Relja Arandjelović and Andrew Zisserman
# IEEE International Conference on Computer Vision (ICCV), Venice, Italy, Oct. 2017.
ast_model = None
ast_processor = None
def get_ast_embeddings(audio_data, sample_rate):
    global ast_model, ast_processor
    if sample_rate != 16000:
        # Resample the audio data to 16 kHz
        audio_data = librosa.resample(y=audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000
    if ast_model is None:
        ast_model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    if ast_processor is None:
        ast_processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    
    inputs = ast_processor(audio_data, sampling_rate=sample_rate, return_tensors="pt")
    with torch.no_grad():
        outputs = ast_model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states.numpy()

# OpenL3
# TODO: Openl3 only supports Python 3.6 to 3.8, while currently we're using Python 3.10 - https://github.com/marl/openl3/issues/96#issuecomment-2077753600
def get_openl3_embeddings(audio_data, sample_rate):
    print('audio_data shape:', audio_data.shape)
    emb, ts = openl3.get_audio_embedding(audio_data, sample_rate)
    # , input_repr="mel128", frontend='librosa' at https://github.com/qdrant/examples/tree/master/qdrant_101_audio_data#openl3
    # other examples at: https://openl3.readthedocs.io/en/latest/tutorial.html
    return emb



# manual / expert features:

def get_weighted_mean_stdv_nomalized( features, energy, sample_rate, normalize_by_nyquist=True ):
    # Compute the weighted mean of the features
    # (weighted by the energy of each frame)
    
    # if energy is all zero values, ensure not dividing by zero
    if np.sum(energy) == 0:
        print('Warning: energy is all zero values')
        return 0, 0
    else:
        weighted_mean_features = np.sum(features * energy) / np.sum(energy)

    # Compute the weighted standard deviation of the features
    # First, computes the deviations from the mean
    deviations = features - weighted_mean_features
    # Weighted variance is the sum of squared deviations times weights over the sum of weights
    weighted_variance_features = np.sum(energy * deviations**2) / np.sum(energy)
    # Standard deviation
    weighted_std_deviation_features = np.sqrt(weighted_variance_features)

    print('Weighted mean of features:', weighted_mean_features)
    print('Weighted standard deviation of features:', weighted_std_deviation_features)

    if np.isnan(weighted_mean_features):
        print('Warning: weighted mean is NaN')

    if normalize_by_nyquist:
        # Compute the max possible feature (approximation)
        # - assume that the feature can range from near 0 to just under half the sample rate (due to the Nyquist-Shannon sampling theorem).
        max_feature = sample_rate / 2
        # Normalize to [0, 1]
        normalized_mean_feature = weighted_mean_features / max_feature
        normalized_std_deviation_feature = weighted_std_deviation_features / max_feature
    else:
        normalized_mean_feature = weighted_mean_features
        normalized_std_deviation_feature = weighted_std_deviation_features

    print('Normalized weighted mean:', normalized_mean_feature)
    print('Normalized weighted standard deviation:', normalized_std_deviation_feature)

    return normalized_mean_feature, normalized_std_deviation_feature

def minmax_normalize(features):
    min_val = np.min(features)
    max_val = np.max(features)
    normalized_features = (features - min_val) / (max_val - min_val)
    return normalized_features

def l1_normalize(vector):
    norm = sum(abs(v) for v in vector)
    return [v / norm for v in vector] if norm != 0 else vector

def l2_normalize(vector):
    norm = sum(v ** 2 for v in vector) ** 0.5
    return [v / norm for v in vector] if norm != 0 else vector

# 3.5.1 Spectral Centroid
def get_spectral_centroid_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01):

    # pyACA
    # [vsc, t] = pyACA.computeFeature("SpectralCentroid", audio_data, sample_rate)
    # vsc_normalized = minmax_normalize(vsc)
    # vsc_mean = np.mean(vsc_normalized)
    # vsc_l2_normalized = l2_normalize(vsc)
    # vsc_l2_normalized_mean = np.mean(vsc_l2_normalized)

    # librosa, weighted by energy
    frame_length_samples = int(frame_length * sample_rate)
    hop_length_samples = int(frame_step * sample_rate)
    # Compute the short-time Fourier transform (STFT) of the audio signal
    stft = np.abs(librosa.stft(audio_data, n_fft=frame_length_samples, hop_length=hop_length_samples))

    # Compute spectral centroids and energy for each frame
    spectral_centroids = librosa.feature.spectral_centroid(S=stft, sr=sample_rate, n_fft=frame_length_samples, hop_length=hop_length_samples)

    # L2 normalization, also called the Euclidean norm
    # spectral_centroids_normalized_sklearn = sklearn.preprocessing.normalize(spectral_centroids, norm='l2')
    # spectral_centroids_normalized_sklearn_mean = np.mean(spectral_centroids_normalized_sklearn)

    spectral_centroids = spectral_centroids[0]
    
    energy = np.sum(stft**2, axis=0)  # Energy of each frame
    spectral_centroids_weighted_mean, spectral_centroids_weighted_stdv = get_weighted_mean_stdv_nomalized(spectral_centroids, energy, sample_rate)
    return spectral_centroids_weighted_mean, spectral_centroids_weighted_stdv


# 3.5.2 Spectral Spread
# There are indications that the spectral spread is related to the timbre of a sound. Sounds with a narrow spectral spread are perceived as bright, while sounds with a wide spectral spread are perceived as dark.
# "There are indications that the spectral spread is of some relevance in describin the perceptual dimensions of timbre" - Lerch
def get_spectral_spread(audio_data, sample_rate):
    [vss, t] = pyACA.computeFeature("SpectralSpread", audio_data, sample_rate)
    vss_l2_normalized = l2_normalize(vss)
    vss_l2_normalized_mean = np.mean(vss_l2_normalized)
    return vss_l2_normalized_mean

# 3.5.3 Spectral Skewness and Kurtosis
# Skewness is a measure of the asymmetry of the distribution of values around the mean. A positive skewness indicates that the distribution is skewed to the right, while a negative skewness indicates that the distribution is skewed to the left.
# Kurtosis is a measure of the "peakedness" of the distribution of values. A high kurtosis indicates that the distribution has a sharp peak, while a low kurtosis indicates that the distribution has a flat peak.
def get_spectral_skewness(audio_data, sample_rate):
    [vsk, t] = pyACA.computeFeature("SpectralSkewness", audio_data, sample_rate)
    vsk_l2_normalized = l2_normalize(vsk)
    vsk_l2_normalized_mean = np.mean(vsk_l2_normalized)
    return vsk_l2_normalized_mean

def get_spectral_kurtosis(audio_data, sample_rate):
    [vsk, t] = pyACA.computeFeature("SpectralKurtosis", audio_data, sample_rate)
    # min-max normalization to get absolute values
    vsk_minmax_normalized = minmax_normalize(vsk)
    vsk_minmax_normalized_mean = np.mean(vsk_minmax_normalized)
    return vsk_minmax_normalized_mean

# 3.5.4 Spectral Rolloff
# The spectral rolloff is a measure of the frequency below which a certain percentage of the total spectral energy is contained. It is related to the brightness of a sound.
def get_spectral_rolloff_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01, roll_percent=0.85):
    stft = np.abs(librosa.stft(audio_data))

    spectral_rolloffs = librosa.feature.spectral_rolloff(S=stft, sr=sample_rate, roll_percent=roll_percent)
    spectral_rolloffs = spectral_rolloffs[0]
    energy = np.sum(stft**2, axis=0)  # Energy of each frame

    return get_weighted_mean_stdv_nomalized(spectral_rolloffs, energy, sample_rate)

# 3.5.5 Spectral Decrease
# "The spectral decrease estimates the steepness of the decrease of the spectral envolope over frequncy." - Lerch
def get_spectral_decrease(audio_data, sample_rate):
    [vsd, t] = pyACA.computeFeature("SpectralDecrease", audio_data, sample_rate)
    vsd_l2_normalized = l2_normalize(vsd)
    vsd_l2_normalized_mean = np.mean(vsd_l2_normalized)
    return vsd_l2_normalized_mean

# 3.5.6 Spectral Slope
def get_spectral_slope(audio_data, sample_rate):
    [vsl, t] = pyACA.computeFeature("SpectralSlope", audio_data, sample_rate)
    vsl_minmax_normalized = minmax_normalize(vsl)
    vsl_minmax_normalized_mean = np.mean(vsl_minmax_normalized)
    return vsl_minmax_normalized_mean

# 3.5.7 Mel Frequency Cepstral Coefficients (MFCCs): for now not concatenated into a single value but rather returned as a feature vector and used in the same way as the (DNN, Transformer) learned features
def get_mfcc_features(audio_data, sample_rate, frame_length=0.025, frame_step=0.01, num_mel_bins=40, num_mfccs=13, lower_frequency=20, upper_frequency=4000):

    # [vmc, t] = pyACA.computeFeature("SpectralMfccs", audio_data, sample_rate)

    """
    Extracts MFCC features from the given audio data.
    Returns a concatenated array of MFCC features for the entire audio data.
    """
    # Extract MFCC features for each frame
    mfcc_features = librosa.feature.mfcc(
        y=audio_data, sr=sample_rate, n_mfcc=num_mfccs, n_mels=num_mel_bins, fmin=lower_frequency, fmax=upper_frequency, hop_length=int(frame_step * sample_rate), n_fft=int(frame_length * sample_rate)
    )
    return mfcc_features

# 3.5.8 Spectral Flux
# The spectral flux is a measure of the change in the spectral envelope of a signal over time.
def get_spectral_flux_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01):

    # pyACA computes a similar value as librosa.onset.onset_strength in this case; opting for it's simpler interface:
    [vsf, t] = pyACA.computeFeature("SpectralFlux", audio_data, sample_rate)
    vsf_l2_normalized = l2_normalize(vsf)
    vsf_l2_normalized_mean = np.mean(vsf_l2_normalized)
    vsf_l2_normalized_stdv = np.std(vsf_l2_normalized)

    # spectral_flux = librosa.onset.onset_strength(y=audio_data, sr=sample_rate, n_fft=int(frame_length * sample_rate), hop_length=int(frame_step * sample_rate))
    # min_spec_flux = np.min(spectral_flux)
    # max_spec_flux = np.max(spectral_flux)
    # normalized_spec_flux = (spectral_flux - min_spec_flux) / (max_spec_flux - min_spec_flux)

    # # spectral_flux = sklearn.preprocessing.normalize(spectral_flux)

    # print('normalized spectral_flux:', normalized_spec_flux)
    # spectral_flux_mean = np.mean(normalized_spec_flux)
    # spectral_flux_stdv = np.std(normalized_spec_flux)
    # return float(spectral_flux_mean), float(spectral_flux_stdv)

    return vsf_l2_normalized_mean, vsf_l2_normalized_stdv

# 3.5.9 Spectral Crest Factor
# The spectral crest factor is a measure of the peakiness of the spectral envelope of a signal.
# "The spectral crest factor gives an estimate on "how sinusodal" a spectrum is. It is a simple measuer of tonalness in the sense that it crudely estimates the amount of tonal components in the signal as opposed to noisy components" - Lerch
def get_spectral_crest_factor(audio_data, sample_rate):
    [vscf, t] = pyACA.computeFeature("SpectralCrestFactor", audio_data, sample_rate)
    # vscf_l2_normalized = l2_normalize(vscf)
    # vscf_l2_normalized_mean = np.mean(vscf_l2_normalized)
    vscf_mean = np.mean(vscf)
    return vscf_mean

# 3.5.10 Spectral Flatness
def get_spectral_flatness_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01):

    # the results from pyACA and librosa differ by orders of magnitude; optiong for pyACA for now

    [vsf, t] = pyACA.computeFeature("SpectralFlatness", audio_data, sample_rate)
    vsf_l2_normalized = l2_normalize(vsf)
    # vsf_l2_normalized_mean = np.mean(vsf_l2_normalized)
    # vsf_l2_normalized_stdv = np.std(vsf_l2_normalized)
    vsf_mean = np.mean(vsf)
    vsf_stdv = np.std(vsf)

    # stft = np.abs(librosa.stft(audio_data))
    # spectral_flatness = librosa.feature.spectral_flatness(S=stft)
    # spectral_flatness = spectral_flatness[0]
    # energy = np.sum(stft**2, axis=0)  # Energy of each frame
    # spectral_flatness_weighted_mean, spectral_flatness_weighted_stdv = get_weighted_mean_stdv_nomalized(spectral_flatness, energy, sample_rate, normalize_by_nyquist=False)
    
    return vsf_mean, vsf_stdv

    # spectral_flatness = librosa.feature.spectral_flatness(y=audio_data, n_fft=int(frame_length * sample_rate), hop_length=int(frame_step * sample_rate))
    # spectral_flatness = spectral_flatness[0]
    # spectral_flatness = sklearn.preprocessing.normalize(spectral_flatness)
    # print('normalized spectral_flatness:', spectral_flatness)
    # spectral_flatness_mean = np.mean(spectral_flatness)
    # spectral_flatness_stdv = np.std(spectral_flatness)
    # return spectral_flatness_mean, spectral_flatness_stdv

# 3.5.11 Tonal Power Ratio
# "Another way to compute the tonalness of a spectrum..." - Lerch
def get_tonal_power_ratio(audio_data, sample_rate):
    [vtp, t] = pyACA.computeFeature("SpectralTonalPowerRatio", audio_data, sample_rate)
    # vtp_l2_normalized = l2_normalize(vtp)
    # vtp_l2_normalized_mean = np.mean(vtp_l2_normalized)
    vtp_mean = np.mean(vtp)
    return vtp_mean
    return vtp_mean

# 3.5.12 Maximum of Autocorrelation Function
def get_max_autocorrelation(audio_data, sample_rate):
    [vmax, t] = pyACA.computeFeature("TimeMaxAcf", audio_data, sample_rate)
    # vmax_l2_normalized = l2_normalize(vmax)
    # vmax_l2_normalized_mean = np.mean(vmax_l2_normalized)
    vmax_mean = np.mean(vmax)
    return vmax_mean

def get_zero_crossing_rate_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01):

    # [vzc, t] = pyACA.computeFeature("TimeZeroCrossingRate", audio_data, sample_rate)
    # vzc_l2_normalized = l2_normalize(vzc)
    # vzc_l2_normalized_mean = np.mean(vzc_l2_normalized)

    frame_length_samples = int(frame_length * sample_rate)
    hop_length_samples = int(frame_step * sample_rate)
    # Compute the short-time Fourier transform (STFT) of the audio signal
    D = librosa.stft(audio_data, n_fft=frame_length_samples, hop_length=hop_length_samples)
    stft = np.abs(D)**2  # Square of the magnitude of the STFT, which gives us the power spectrum

    # Compute Energy of the STFT for each frame
    energy = np.sum(stft, axis=0)

    # Compute Zero Crossing Rate
    zero_crossing_rates = librosa.feature.zero_crossing_rate(audio_data, frame_length=frame_length_samples, hop_length=hop_length_samples)
    zero_crossing_rates = zero_crossing_rates[0]

    zero_crossing_rate_weighted_mean, zero_crossing_rate_weighted_stdv = get_weighted_mean_stdv_nomalized(zero_crossing_rates, energy, sample_rate, normalize_by_nyquist=False)
    return zero_crossing_rate_weighted_mean, zero_crossing_rate_weighted_stdv
    

def get_chroma_stft_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01):
    chroma_stft = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate, n_fft=int(frame_length * sample_rate), hop_length=int(frame_step * sample_rate))
    chroma_stft = chroma_stft[0]
    # chroma_stft = sklearn.preprocessing.normalize(chroma_stft)
    # print('normalized chroma_stft:', chroma_stft)
    chroma_stft_mean = float(np.mean(chroma_stft)) # convert to float for JSON serialization
    chroma_stft_stdv = float(np.std(chroma_stft))
    print('chroma_stft_mean:', chroma_stft_mean)
    print('chroma_stft_stdv:', chroma_stft_stdv)
    return chroma_stft_mean, chroma_stft_stdv

# TODO ommit
def get_mel_spectrogram_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01, num_mel_bins=40):
    # mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=num_mel_bins, n_fft=int(frame_length * sample_rate), hop_length=int(frame_step * sample_rate))
    # mel_spectrogram = mel_spectrogram[0]
    # mel_spectrogram = sklearn.preprocessing.normalize(mel_spectrogram)
    # print('normalized mel_spectrogram:', mel_spectrogram)
    # mel_spectrogram_mean = np.mean(mel_spectrogram)
    # mel_spectrogram_stdv = np.std(mel_spectrogram)
    # return mel_spectrogram_mean, mel_spectrogram_stdv

    # Compute the mel spectrogram
    S = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=num_mel_bins, n_fft=int(frame_length * sample_rate), hop_length=int(frame_step * sample_rate))

    # It's important to note here that we use the power specification (power=2) when calling melspectrogram.
    # This gives us the power in each mel band and time frame.

    # Convert the power mel spectrogram to decibel (dB) units
    S_db = librosa.power_to_db(S, ref=np.max)

    # Compute the overall weighted mean where the weights are the total energy in each time frame
    energy_per_frame = np.sum(S, axis=0)  # Sum power across mel bands
    overall_energy = np.sum(energy_per_frame)  # Sum energy across all frames
    weighted_mean_db = np.sum(S_db * energy_per_frame) / overall_energy

    # Compute the overall weighted standard deviation
    weighted_std_dev_db = np.sqrt(np.sum((S_db - weighted_mean_db) ** 2 * energy_per_frame) / overall_energy)

    print('Overall weighted mean of the mel spectrogram (dB):', weighted_mean_db)
    print('Overall weighted standard deviation of the mel spectrogram (dB):', weighted_std_dev_db)

    min_db = -80  # Minimum expected dB value
    max_db = 0    # Maximum dB value (0dB being the reference power)

    normalized_mean = (weighted_mean_db - min_db) / (max_db - min_db)
    normalized_std_dev = weighted_std_dev_db / (max_db - min_db)

    normalized_mean = np.clip(normalized_mean, 0, 1)
    normalized_std_dev = np.clip(normalized_std_dev, 0, 1)

    print('Normalized overall weighted mean:', normalized_mean)
    print('Normalized overall weighted standard deviation:', normalized_std_dev)




def get_rms_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01):
    # rms = librosa.feature.rms(y=audio_data, frame_length=int(frame_length * sample_rate), hop_length=int(frame_step * sample_rate))
    # rms = rms[0]
    # rms = sklearn.preprocessing.normalize(rms)
    # print('normalized rms:', rms)
    # rms_mean = np.mean(rms)
    # rms_stdv = np.std(rms)
    # return rms_mean, rms_stdv

    # Normalize the waveform to be between -1 and 1
    audio_data /= np.max(np.abs(audio_data))

    # Compute RMS energy
    rmse = librosa.feature.rms(y=audio_data, frame_length=int(frame_length * sample_rate), hop_length=int(frame_step * sample_rate))

    # Compute weighted mean RMS energy (using linear scale, not dB)
    mean_rmse_weighted = np.sum(rmse**2) / np.sum(rmse)

    # Compute weighted standard deviation RMS energy (using linear scale, not dB)
    std_dev_rmse_weighted = np.sqrt(np.sum(((rmse - mean_rmse_weighted)**2) * rmse) / np.sum(rmse))

    print('Weighted mean RMS energy:', mean_rmse_weighted)
    print('Weighted standard deviation of RMS energy:', std_dev_rmse_weighted)

    return float(mean_rmse_weighted), float(std_dev_rmse_weighted)

def get_spectral_bandwidth_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01):
    frame_length_samples = int(frame_length * sample_rate)
    hop_length_samples = int(frame_step * sample_rate)

    spectral_bandwidths = librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate, n_fft=frame_length_samples, hop_length=hop_length_samples)
    spectral_bandwidths = spectral_bandwidths[0]

    stft = np.abs(librosa.stft(audio_data, n_fft=frame_length_samples, hop_length=hop_length_samples))
    energy = np.sum(stft**2, axis=0)  # Energy of each frame

    return get_weighted_mean_stdv_nomalized(spectral_bandwidths, energy, sample_rate)

    # spectral_bandwidths = sklearn.preprocessing.normalize(spectral_bandwidths)
    # print('normalized spectral_bandwidths:', spectral_bandwidths)
    # spectral_bandwidth_mean = np.mean(spectral_bandwidths)
    # spectral_bandwidth_stdv = np.std(spectral_bandwidths)
    # return spectral_bandwidth_mean, spectral_bandwidth_stdv

def get_spectral_contrast_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01, num_bands=6):
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate, n_bands=num_bands, n_fft=int(frame_length * sample_rate), hop_length=int(frame_step * sample_rate))
    spectral_contrast = spectral_contrast[0]
    # spectral_contrast = sklearn.preprocessing.normalize(spectral_contrast)
    normalized_spectral_contrast = (spectral_contrast - np.min(spectral_contrast)) / (np.max(spectral_contrast) - np.min(spectral_contrast))
    print('normalized spectral_contrast:', normalized_spectral_contrast)
    spectral_contrast_mean = np.mean(normalized_spectral_contrast)
    spectral_contrast_stdv = np.std(normalized_spectral_contrast)
    return spectral_contrast_mean, spectral_contrast_stdv


# Inspired by: https://doi.org/10.3390/app112411926 :
# "... basic statistics (mean, standard deviation, minimum and maximum), which are applied to both the raw features and their first derivative. 
# For example, for thirteen dimensions in the frame-level feature, this renders a vector of 104 dimensions."
def compute_feature_statistics(feature_matrix):
    """
    Compute the mean, standard deviation, minimum, and maximum for each feature dimension
    and for the first derivatives of the feature dimensions. Returns a combined 
    statistics vector of size (num_features * 8).

    :param feature_matrix: A NumPy array with shape (num_features, num_time_steps).
    :return: A NumPy array containing all combined statistics.
    """
    num_features, num_time_steps = feature_matrix.shape
    
    # Compute the statistics for the feature values
    means = np.mean(feature_matrix, axis=1)
    std_devs = np.std(feature_matrix, axis=1, ddof=1)
    minimums = np.min(feature_matrix, axis=1)
    maximums = np.max(feature_matrix, axis=1)
    
    # Calculate the first derivatives along the time axis
    derivatives = np.diff(feature_matrix, axis=1)
    
    # Compute the statistics for the first derivatives
    der_means = np.mean(derivatives, axis=1)
    der_std_devs = np.std(derivatives, axis=1, ddof=1)
    der_minimums = np.min(derivatives, axis=1)
    der_maximums = np.max(derivatives, axis=1)
    
    # Combine the statistics into a single vector
    statistics_vector = np.concatenate((means, std_devs, minimums, maximums,
                                        der_means, der_std_devs, der_minimums, der_maximums))
    
    return statistics_vector


# TODO special iteration on structural features
def get_tempo(audio_data, sample_rate):
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
    print('tempo:', tempo)
    return tempo