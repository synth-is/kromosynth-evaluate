import librosa
import numpy as np

# idea from: https://medium.com/@LeonFedden/comparative-audio-analysis-with-wavenet-mfccs-umap-t-sne-and-pca-cb8237bfce2f
def get_mfcc_feature_means_stdv_firstorderdifference_concatenated(audio_data, sample_rate, frame_length=0.025, frame_step=0.01, num_mel_bins=40, num_mfccs=13, lower_frequency=20, upper_frequency=4000):
    """
    Extracts MFCC features from the given audio data.
    Returns a concatenated array of MFCC features for the entire audio data.
    """
    # Extract MFCC features for each frame
    mfcc_features = librosa.feature.mfcc(
        y=audio_data, sr=sample_rate, n_mfcc=num_mfccs, n_mels=num_mel_bins, fmin=lower_frequency, fmax=upper_frequency, hop_length=int(frame_step * sample_rate), n_fft=int(frame_length * sample_rate)
    )

    # print('MFCC features shape:', mfcc_features.shape)
    # print('MFCC features:', mfcc_features)
    # print first 5 frames
    # print('First 5 frames:', mfcc_features[:, 0:5])

    # Transpose the array so that the shape is (num_mfccs, num_frames)
    mfcc_features = mfcc_features.T

    # Calculate the mean and standard deviation of each MFCC feature
    mfcc_feature_means = np.mean(mfcc_features, axis=0)
    mfcc_feature_stdv = np.std(mfcc_features, axis=0)

    # # Calculate the first-order difference of each MFCC feature
    # mfcc_feature_firstorderdifference = np.diff(mfcc_features, axis=0)

    # Get the average difference of the features
    average_difference_features = np.zeros((num_mfccs,))
    for i in range(0, len(mfcc_features) - 2, 2):
        average_difference_features += mfcc_features[i] - mfcc_features[i+1]
    average_difference_features /= (len(mfcc_features) // 2)   
    average_difference_features = np.array(average_difference_features)

    # Concatenate the MFCC features for each frame into a single array
    mfcc_features_concatenated = np.hstack((mfcc_feature_means, mfcc_feature_stdv, average_difference_features))

    return mfcc_features_concatenated