import librosa
import numpy as np
from frechet_audio_distance import FrechetAudioDistance
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


def get_mfcc_features(audio_data, sample_rate, frame_length=0.025, frame_step=0.01, num_mel_bins=40, num_mfccs=13, lower_frequency=20, upper_frequency=4000):
    """
    Extracts MFCC features from the given audio data.
    Returns a concatenated array of MFCC features for the entire audio data.
    """
    # Extract MFCC features for each frame
    mfcc_features = librosa.feature.mfcc(
        y=audio_data, sr=sample_rate, n_mfcc=num_mfccs, n_mels=num_mel_bins, fmin=lower_frequency, fmax=upper_frequency, hop_length=int(frame_step * sample_rate), n_fft=int(frame_length * sample_rate)
    )
    return mfcc_features

# the following functions could in fact be just one, but here they're separate for some kind of documentation

# sample_rate should be 16000 for VGGish
def get_vggish_embeddings(audio_data, sample_rate, ckpt_dir, use_pca=False, use_activation=False):
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

# sample_rate should be 8000, 16000 or 32000 for PANN
def get_pann_embeddings(audio_data, sample_rate, ckpt_dir):
    frechet = FrechetAudioDistance(
        ckpt_dir=ckpt_dir,
        model_name="pann",
        sample_rate=sample_rate,
        verbose=True,
        audio_load_worker=8,
    )
    embeddings = frechet.get_embeddings(audio_data, sample_rate)
    return embeddings

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
    frechet = FrechetAudioDistance(
        ckpt_dir=ckpt_dir,
        model_name="encodec",
        sample_rate=sample_rate,
        verbose=True,
        audio_load_worker=8,
    )
    embeddings = frechet.get_embeddings(audio_data, sample_rate)
    return embeddings


def get_weighted_mean_stdv_nomalized( features, energy, sample_rate, normalize_by_nyquist=True ):
    # Compute the weighted mean of the features
    # (weighted by the energy of each frame)
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
    

def get_spectral_centroid_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01):
    frame_length_samples = int(frame_length * sample_rate)
    hop_length_samples = int(frame_step * sample_rate)
    # Compute the short-time Fourier transform (STFT) of the audio signal
    stft = np.abs(librosa.stft(audio_data, n_fft=frame_length_samples, hop_length=hop_length_samples))

    # Compute spectral centroids and energy for each frame
    spectral_centroids = librosa.feature.spectral_centroid(S=stft, sr=sample_rate, n_fft=frame_length_samples, hop_length=hop_length_samples)
    spectral_centroids = spectral_centroids[0]
    
    energy = np.sum(stft**2, axis=0)  # Energy of each frame

    return get_weighted_mean_stdv_nomalized(spectral_centroids, energy, sample_rate)

def get_spectral_rolloff_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01, roll_percent=0.85):
    stft = np.abs(librosa.stft(audio_data))

    spectral_rolloffs = librosa.feature.spectral_rolloff(S=stft, sr=sample_rate, roll_percent=roll_percent)
    spectral_rolloffs = spectral_rolloffs[0]
    energy = np.sum(stft**2, axis=0)  # Energy of each frame

    return get_weighted_mean_stdv_nomalized(spectral_rolloffs, energy, sample_rate)

def get_zero_crossing_rate_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01):
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

    return get_weighted_mean_stdv_nomalized(zero_crossing_rates, energy, sample_rate, normalize_by_nyquist=False)

def get_tempo(audio_data, sample_rate):
    tempo, _ = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
    print('tempo:', tempo)
    return tempo

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

def get_spectral_flatness_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01):
    stft = np.abs(librosa.stft(audio_data))
    spectral_flatness = librosa.feature.spectral_flatness(S=stft)
    spectral_flatness = spectral_flatness[0]
    energy = np.sum(stft**2, axis=0)  # Energy of each frame

    return get_weighted_mean_stdv_nomalized(spectral_flatness, energy, sample_rate, normalize_by_nyquist=False)

    # spectral_flatness = librosa.feature.spectral_flatness(y=audio_data, n_fft=int(frame_length * sample_rate), hop_length=int(frame_step * sample_rate))
    # spectral_flatness = spectral_flatness[0]
    # spectral_flatness = sklearn.preprocessing.normalize(spectral_flatness)
    # print('normalized spectral_flatness:', spectral_flatness)
    # spectral_flatness_mean = np.mean(spectral_flatness)
    # spectral_flatness_stdv = np.std(spectral_flatness)
    # return spectral_flatness_mean, spectral_flatness_stdv

def get_spectral_flux_mean_stdv(audio_data, sample_rate, frame_length=0.025, frame_step=0.01):
    spectral_flux = librosa.onset.onset_strength(y=audio_data, sr=sample_rate, n_fft=int(frame_length * sample_rate), hop_length=int(frame_step * sample_rate))
    min_spec_flux = np.min(spectral_flux)
    max_spec_flux = np.max(spectral_flux)
    normalized_spec_flux = (spectral_flux - min_spec_flux) / (max_spec_flux - min_spec_flux)

    # spectral_flux = sklearn.preprocessing.normalize(spectral_flux)

    print('normalized spectral_flux:', normalized_spec_flux)
    spectral_flux_mean = np.mean(normalized_spec_flux)
    spectral_flux_stdv = np.std(normalized_spec_flux)
    return float(spectral_flux_mean), float(spectral_flux_stdv)   
