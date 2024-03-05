import librosa
import numpy as np
from frechet_audio_distance import FrechetAudioDistance

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