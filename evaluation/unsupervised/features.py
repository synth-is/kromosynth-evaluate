# 1: websocket server to extract MFCC features from audio data

import asyncio
import websockets
import json
import argparse
import numpy as np
import os
from setproctitle import setproctitle
import time
from urllib.parse import urlparse, parse_qs

# import the function get_mfcc_feature_means_stdv_firstorderdifference_concatenated from measurements/diversity/mfcc.py
import sys
sys.path.append('../..')
from measurements.diversity.audio_features import (
    get_feature_means_stdv_firstorderdifference_concatenated,
    get_mfcc_features, 
    get_vggish_embeddings, 
    get_vggish_embeddings_essentia,
    # get_pann_embeddings, get_pann_embeddings_panns_inference,
    # get_clap_embeddings, get_encodec_embeddings,
    get_discogs_effnet_embeddings,
    get_maest_embeddings, get_msd_musicnn_embeddings,
    get_wav2vec_embeddings,
    get_ast_embeddings,
    get_openl3_embeddings,
    get_spectral_centroid_mean_stdv, 
    get_spectral_spread, get_spectral_skewness, get_spectral_kurtosis, get_spectral_decrease, get_spectral_slope, get_spectral_crest_factor, get_tonal_power_ratio, get_max_autocorrelation,
    get_spectral_rolloff_mean_stdv, get_zero_crossing_rate_mean_stdv, get_tempo,
    get_chroma_stft_mean_stdv, get_mel_spectrogram_mean_stdv, get_rms_mean_stdv, get_spectral_bandwidth_mean_stdv,
    get_spectral_contrast_mean_stdv, get_spectral_flatness_mean_stdv, get_spectral_flux_mean_stdv, get_spectral_rolloff_mean_stdv,
    compute_feature_statistics
)
from evaluation.util import filepath_to_port

async def socket_server(websocket, path):
    # Parse the path and query components from the URL
    url_components = urlparse(path)
    request_path = url_components.path
    query_params = parse_qs(url_components.query)
    try:
        sample_rate = int(query_params.get('sample_rate', [16000])[0])
        # Get MFCC focus parameter
        mfcc_focus = query_params.get('mfcc_focus', [None])[0]
        audio_sub_region = query_params.get('audio_sub_region', [None])[0] # e.g. "3/4" or "1/2"; the former would mean take the last quarter of the audio data, the latter would mean take the first half of the audio data
        
        # Validate mfcc_focus if provided
        if mfcc_focus and mfcc_focus not in ['timbre', 'spectral', 'energy', 'temporal', 'full']:
            mfcc_focus = 'full'  # Default to full if invalid
            
        # Wait for the first message and determine its type
        message = await websocket.recv()

        if isinstance(message, bytes):
            start = time.time()
            # Received binary message (assume it's an audio buffer)
            audio_data = message
            print('Audio data received for feature extraction')
            # convert the audio data to a numpy array
            audio_data = np.frombuffer(audio_data, dtype=np.float32)
            # If audio_sub_region is set, select the specified portion
            if audio_sub_region:
                try:
                    num, denom = map(int, audio_sub_region.split('/'))
                    if denom > 0 and 0 < num <= denom:
                        total_len = len(audio_data)
                        region_len = total_len * num // denom
                        if num == denom:
                            print(f"audio_sub_region '{audio_sub_region}': using full audio ({total_len} samples)")
                            # Take the whole audio
                            pass
                        elif num < denom:
                            start = total_len - region_len
                            print(f"audio_sub_region '{audio_sub_region}': using last {region_len} samples (from {start} to {total_len})")
                            audio_data = audio_data[start:]
                        else:
                            print(f"audio_sub_region '{audio_sub_region}': num > denom, using full audio ({total_len} samples)")
                            # Fallback: just use the whole audio
                            pass
                    else:
                        print(f"audio_sub_region '{audio_sub_region}': invalid num/denom values, using full audio")
                except Exception as e:
                    print(f"Invalid audio_sub_region '{audio_sub_region}': {e}")
            # Process the audio data...

            if request_path == '/mfcc' or request_path == '/':
                print('Extracting MFCC features...')
                embeddings = get_mfcc_features(audio_data, sample_rate)
                features = get_feature_means_stdv_firstorderdifference_concatenated(embeddings)
                features_type = 'mfcc'
                
                # Apply focus weights if specified
                if mfcc_focus:
                    print(f'Applying {mfcc_focus} focus to MFCC features...')
                    features = apply_focus_to_feature_vector(features, mfcc_focus, features_type)
                    features_type = f'mfcc-{mfcc_focus}'
                    
            elif request_path == '/mfcc-sans0': # without the first coefficient
                print('Extracting MFCC features without the first coefficient...')
                embeddings = get_mfcc_features(audio_data, sample_rate)
                embeddings = embeddings[1:]
                features = get_feature_means_stdv_firstorderdifference_concatenated(embeddings)
                features_type = 'mfcc-sans0'
                
                # Apply focus weights if specified
                if mfcc_focus:
                    print(f'Applying {mfcc_focus} focus to MFCC-sans0 features...')
                    features = apply_focus_to_feature_vector(features, mfcc_focus, features_type)
                    features_type = f'mfcc-sans0-{mfcc_focus}'
                    
            elif request_path == '/mfcc-statistics':
                print('Extracting MFCC statistics...')
                embeddings = get_mfcc_features(audio_data, sample_rate)
                features = compute_feature_statistics(embeddings)
                features_type = 'mfcc-statistics'
                
                # Apply focus weights if specified
                if mfcc_focus:
                    print(f'Applying {mfcc_focus} focus to MFCC statistics...')
                    features = apply_focus_to_feature_vector(features, mfcc_focus, features_type)
                    features_type = f'mfcc-statistics-{mfcc_focus}'
                    
            elif request_path == '/mfcc-sans0-statistics':
                print('Extracting MFCC statistics without the first coefficient...')
                embeddings = get_mfcc_features(audio_data, sample_rate)
                embeddings = embeddings[1:]
                features = compute_feature_statistics(embeddings)
                features_type = 'mfcc-sans0-statistics'
                
                # Apply focus weights if specified
                if mfcc_focus:
                    print(f'Applying {mfcc_focus} focus to MFCC-sans0 statistics...')
                    features = apply_focus_to_feature_vector(features, mfcc_focus, features_type)
                    features_type = f'mfcc-sans0-statistics-{mfcc_focus}'
                    
            else:
                ckpt_dir = query_params.get('ckpt_dir', [None])[0]
                # if ckpt_dir contains "/localscratch/<job-ID>" then replace the job-ID with the environment variable SLURM_JOB_ID
                if ckpt_dir is not None and '/localscratch/' in ckpt_dir:
                    ckpt_dir = ckpt_dir.replace('/localscratch/<job-ID>', '/localscratch/' + os.environ.get('SLURM_JOB_ID') )
                print('ckpt_dir:', ckpt_dir)
                if request_path == '/vggish':
                    audio_data = [audio_data]
                    print('Extracting VGGish embeddings...')
                    use_pca = query_params.get('use_pca', [False])[0]
                    use_activation = query_params.get('use_activation', [False])[0]
                    embeddings = get_vggish_embeddings(audio_data, sample_rate, ckpt_dir, use_pca, use_activation)
                    # features = get_feature_means_stdv_firstorderdifference_concatenated(embeddings.T) # transpose the embeddings to match the shape of the MFCC features: (num_features, num_frames
                    features = np.mean(embeddings, axis=0)
                    features_type = 'vggish'
                elif request_path == '/vggishessentia':
                    print('Extracting VGGish embeddings using Essentia...')
                    embeddings = get_vggish_embeddings_essentia(audio_data, sample_rate, MODELS_PATH)
                    # features = get_feature_means_stdv_firstorderdifference_concatenated(embeddings.T)
                    features = np.mean(embeddings, axis=0)
                    features_type = 'vggish-essentia'
                elif request_path == '/pann':
                    print('Extracting PANN embeddings...')
                    audio_data = [audio_data]
                    embeddings = get_pann_embeddings(audio_data, sample_rate, ckpt_dir)
                    features = embeddings
                    features_type = 'pann'
                elif request_path == '/panns-inference':
                    print('Extracting PANN embeddings using inference...')
                    embeddings = get_pann_embeddings_panns_inference(audio_data, sample_rate, ckpt_dir+'/panns_data/Cnn14_mAP=0.431.pth')
                    features = embeddings[0]
                    features_type = 'panns-inference'
                elif request_path == '/clap':
                    print('Extracting CLAP embeddings...')
                    audio_data = [audio_data]
                    submodel_name = query_params.get('submodel_name', ["630k-audioset"])[0]
                    enable_fusion = query_params.get('enable_fusion', [False])[0]
                    embeddings = get_clap_embeddings(audio_data, sample_rate, ckpt_dir, submodel_name, enable_fusion)
                    features = embeddings[0]
                    features_type = 'clap'
                elif request_path == '/encodec':
                    print('Extracting EnCodec embeddings...')
                    audio_data = [audio_data]
                    embeddings = get_encodec_embeddings(audio_data, sample_rate, ckpt_dir)
                    features = get_feature_means_stdv_firstorderdifference_concatenated(embeddings)
                    features_type = 'encodec'
                elif request_path == '/maest':
                    print('Extracting MAEST embeddings...')
                    audio_data = audio_data
                    embeddings = get_maest_embeddings(audio_data, sample_rate, MODELS_PATH)
                    features = np.mean(embeddings[0][0], axis=0)
                    features_type = 'maest'
                elif request_path == '/discogs-effnet':
                    print('Extracting Discogs EfficientNet embeddings...')
                    embeddings = get_discogs_effnet_embeddings(audio_data, sample_rate, MODELS_PATH)
                    features = np.mean(embeddings, axis=0)
                    features_type = 'discogs-effnet'
                elif request_path == '/msd-musicnn':
                    print('Extracting MSD Musicnn embeddings...')
                    embeddings = get_msd_musicnn_embeddings(audio_data, sample_rate, MODELS_PATH)
                    features = np.mean(embeddings, axis=0)
                    features_type = 'msd-musicnn'
                elif request_path == '/wav2vec':
                    print('Extracting Wav2Vec embeddings...')
                    embeddings = get_wav2vec_embeddings(audio_data, sample_rate)
                    features = embeddings[0]
                    features_type = 'wav2vec'
                elif request_path == '/ast':
                    print('Extracting AST embeddings...')
                    embeddings = get_ast_embeddings(audio_data, sample_rate)
                    features = np.mean(embeddings[0], axis=0)
                    features_type = 'ast'
                # Openl3 only supports Python 3.6 to 3.8, while currently we're using Python 3.10 - https://github.com/marl/openl3/issues/96#issuecomment-2077753600
                # elif request_path == '/openl3':
                #     print('Extracting OpenL3 embeddings...')
                #     embeddings = get_openl3_embeddings(audio_data, sample_rate)
                #     features = np.mean(embeddings, axis=0)
                #     features_type = 'openl3'
                elif request_path == '/manual':
                    # Manually defined extraction of instantaneous features
                    print('Extracting manually defined features...')
                    audio_data = np.array(audio_data)
                    embeddings = np.array([]) # just to keep the code running
                    # array of manually defined features names from a comma separated string in the query parameter
                    features_type = query_params.get('features', [''])[0]
                    feature_names = features_type.split(',')
                    features = []
                    features_stdv = []
                    
                    # add each feature to the features array
                    for feature_name in feature_names:
                        if feature_name == 'spectral_centroid':
                            spectral_centroids_mean, spectral_centroids_stdv = get_spectral_centroid_mean_stdv(audio_data, sample_rate)
                            features.append(spectral_centroids_mean)
                            features_stdv.append(spectral_centroids_stdv)
                        elif feature_name == 'spectral_spread':
                            spectral_spread = get_spectral_spread(audio_data, sample_rate)
                            features.append(spectral_spread)
                        elif feature_name == 'spectral_skewness':
                            spectral_skewness = get_spectral_skewness(audio_data, sample_rate)
                            features.append(spectral_skewness)
                        elif feature_name == 'spectral_kurtosis':
                            spectral_kurtosis = get_spectral_kurtosis(audio_data, sample_rate)
                            features.append(spectral_kurtosis)
                        elif feature_name == 'spectral_rolloff':
                            spectral_rolloff_mean, spectral_rolloff_stdv = get_spectral_rolloff_mean_stdv(audio_data, sample_rate)
                            features.append(spectral_rolloff_mean)
                            features_stdv.append(spectral_rolloff_stdv)
                        elif feature_name == 'spectral_decrease':
                            spectral_decrease = get_spectral_decrease(audio_data, sample_rate)
                            features.append(spectral_decrease)
                        elif feature_name == 'spectral_slope':
                            spectral_slope = get_spectral_slope(audio_data, sample_rate)
                            features.append(spectral_slope)
                        elif feature_name == 'spectral_flux':
                            spectral_flux_mean, spectral_flux_stdv = get_spectral_flux_mean_stdv(audio_data, sample_rate)
                            features.append(spectral_flux_mean)
                            features_stdv.append(spectral_flux_stdv)
                        elif feature_name == 'spectral_crest_factor':
                            spectral_crest_factor = get_spectral_crest_factor(audio_data, sample_rate)
                            features.append(spectral_crest_factor)
                        elif feature_name == 'spectral_flatness':
                            spectral_flatness_mean, spectral_flatness_stdv = get_spectral_flatness_mean_stdv(audio_data, sample_rate)
                            features.append(spectral_flatness_mean)
                            features_stdv.append(spectral_flatness_stdv)
                        elif feature_name == 'tonal_power_ratio':
                            tonal_power_ratio = get_tonal_power_ratio(audio_data, sample_rate)
                            features.append(tonal_power_ratio)
                        elif feature_name == 'max_autocorrelation':
                            max_autocorrelation = get_max_autocorrelation(audio_data, sample_rate)
                            features.append(max_autocorrelation)
                        elif feature_name == 'zero_crossing_rate':
                            zero_crossing_rate, zero_crossing_rate_stdv = get_zero_crossing_rate_mean_stdv(audio_data, sample_rate)
                            features.append(zero_crossing_rate)
                            features_stdv.append(zero_crossing_rate_stdv)

                        elif feature_name == 'chroma_stft':
                            chrome_stft_mean, chroma_stft_stdv = get_chroma_stft_mean_stdv(audio_data, sample_rate)
                            features.append(chrome_stft_mean)
                            features_stdv.append(chroma_stft_stdv)
                        # elif feature_name == 'mel_spectrogram':
                        #     mel_spectrogram_mean, mel_spectrogram_stdv = get_mel_spectrogram_mean_stdv(audio_data, sample_rate)
                        #     features.append(mel_spectrogram_mean)
                        #     features_stdv.append(mel_spectrogram_stdv)
                        elif feature_name == 'rms':
                            rms_mean, rms_stdv = get_rms_mean_stdv(audio_data, sample_rate)
                            features.append(rms_mean)
                            features_stdv.append(rms_stdv)
                        elif feature_name == 'spectral_bandwidth':
                            spectral_bandwidth, spectral_bandwidth_stdv = get_spectral_bandwidth_mean_stdv(audio_data, sample_rate)
                            features.append(spectral_bandwidth)
                            features_stdv.append(spectral_bandwidth_stdv)
                        elif feature_name == 'spectral_contrast':
                            spectral_contrast_mean, spectral_contrast_stdv = get_spectral_contrast_mean_stdv(audio_data, sample_rate)
                            features.append(spectral_contrast_mean)
                            features_stdv.append(spectral_contrast_stdv)

                        elif feature_name == 'tempo':
                            tempo_mean, tempo_stdv = get_tempo(audio_data, sample_rate)
                            features.append(tempo_mean)
                            features_stdv.append(tempo_stdv)


                            
            print('Embeddings extracted shape:', embeddings.shape)

            # features = get_feature_means_stdv_firstorderdifference_concatenated(embeddings)

            print('Features extracted shape:', np.array(features).shape)

            # print('features:', features)

            # print('MFCC features extracted:', features)

            end = time.time()
            print('features: Time taken to extract features:', end - start)

            # if features is a numpy array, convert it to a list
            if isinstance(features, np.ndarray):
                features = features.tolist()

            # ensure all values in features are of type float
            features = [float(f) for f in features]

            features = np.nan_to_num(features).tolist()

            response = {'status': 'received standalone audio', 'features': features, 'embedding': embeddings.tolist(), 'type': features_type, 'time': end - start}
            await websocket.send(json.dumps(response))
            await asyncio.sleep(30)
    except websockets.ConnectionClosed as e:
        print('features: ConnectionClosed', e)
        response = {'status': 'ERROR' + str(e)}
        await websocket.send(json.dumps(response))
    except Exception as e:
        print('features: Exception', e)
        response = {'status': 'ERROR' + str(e)}
        await websocket.send(json.dumps(response))



def get_feature_vector_focus_indices(feature_type, focus_type):
    """
    Get the appropriate feature vector indices for different MFCC focus types.
    
    Parameters:
    -----------
    feature_type : str
        The type of MFCC feature extraction used ('mfcc', 'mfcc-sans0', 'mfcc-statistics', 'mfcc-sans0-statistics')
    focus_type : str
        Type of focus: 'timbre', 'spectral', 'energy', 'temporal', 'full'
    
    Returns:
    --------
    indices : list
        List of indices to select from the feature vector
    """
    # Determine if we're using statistics-based features
    is_statistics = 'statistics' in feature_type
    # Determine if we're using sans0 features
    is_sans0 = 'sans0' in feature_type
    
    # Number of coefficients
    n_coeffs = 12 if is_sans0 else 13
    
    # For non-statistics features (standard MFCC)
    if not is_statistics:
        # Format: [means(13), std_devs(13), first_order_diffs(13)]
        # Or for sans0: [means(12), std_devs(12), first_order_diffs(12)]
        
        if focus_type == 'timbre':
            # For timbre focus, use coefficients 1-5 (indices 1-5 or 0-4 for sans0)
            start_idx = 1 if not is_sans0 else 0
            # Indices for means
            indices = list(range(start_idx, start_idx + 5))
            # Add indices for std_devs and first_order_diffs
            indices += [i + n_coeffs for i in indices]
            indices += [i + 2*n_coeffs for i in indices[:len(indices)]]
            return indices
            
        elif focus_type == 'spectral':
            # For spectral focus, use coefficients 6-12 (indices 6-12 or 5-11 for sans0)
            start_idx = 6 if not is_sans0 else 5
            end_idx = n_coeffs
            # Indices for means
            indices = list(range(start_idx, end_idx))
            # Add indices for std_devs and first_order_diffs
            indices += [i + n_coeffs for i in indices]
            indices += [i + 2*n_coeffs for i in indices[:len(indices)]]
            return indices
            
        elif focus_type == 'energy':
            if is_sans0:
                # Energy focus not available for sans0 features
                return list(range(3*n_coeffs))  # Return all indices
            # For energy focus, use coefficient 0 (index 0)
            indices = [0]
            # Add indices for std_dev and first_order_diff
            indices += [n_coeffs]
            indices += [2*n_coeffs]
            return indices
            
        elif focus_type == 'temporal':
            # For temporal focus, use all first_order_diffs
            return list(range(2*n_coeffs, 3*n_coeffs))
            
        elif focus_type == 'full':
            # Use all coefficients
            return list(range(3*n_coeffs))
            
    else:
        # For statistics-based features 
        # Format: [means(13), std_devs(13), mins(13), maxes(13), der_means(13), der_std_devs(13), der_mins(13), der_maxes(13)]
        # Or for sans0 with 12 coefficients instead of 13
        
        if focus_type == 'timbre':
            # Coefficients 1-5 (indices 1-5 or 0-4 for sans0)
            start_idx = 1 if not is_sans0 else 0
            timbre_indices = list(range(start_idx, start_idx + 5))
            
            # Gather all related statistics for these coefficients
            indices = []
            for i in timbre_indices:
                # Add mean, std_dev, min, max
                indices.append(i)
                indices.append(i + n_coeffs)
                indices.append(i + 2*n_coeffs)
                indices.append(i + 3*n_coeffs)
                # Add derivative statistics
                indices.append(i + 4*n_coeffs)
                indices.append(i + 5*n_coeffs)
                indices.append(i + 6*n_coeffs)
                indices.append(i + 7*n_coeffs)
            return indices
            
        elif focus_type == 'spectral':
            # Coefficients 6-12 (indices 6-12 or 5-11 for sans0)
            start_idx = 6 if not is_sans0 else 5
            end_idx = n_coeffs
            spectral_indices = list(range(start_idx, end_idx))
            
            # Gather all related statistics for these coefficients
            indices = []
            for i in spectral_indices:
                # Add mean, std_dev, min, max
                indices.append(i)
                indices.append(i + n_coeffs)
                indices.append(i + 2*n_coeffs)
                indices.append(i + 3*n_coeffs)
                # Add derivative statistics
                indices.append(i + 4*n_coeffs)
                indices.append(i + 5*n_coeffs)
                indices.append(i + 6*n_coeffs)
                indices.append(i + 7*n_coeffs)
            return indices
            
        elif focus_type == 'energy':
            if is_sans0:
                # Energy focus not available for sans0 features
                return list(range(8*n_coeffs))  # Return all indices
                
            # For energy focus, use coefficient 0 (index 0)
            i = 0
            indices = [
                i, i + n_coeffs, i + 2*n_coeffs, i + 3*n_coeffs,  # Mean, std_dev, min, max
                i + 4*n_coeffs, i + 5*n_coeffs, i + 6*n_coeffs, i + 7*n_coeffs  # Derivative statistics
            ]
            return indices
            
        elif focus_type == 'temporal':
            # For temporal focus, use all derivative statistics
            return list(range(4*n_coeffs, 8*n_coeffs))
            
        elif focus_type == 'full':
            # Use all coefficients and statistics
            return list(range(8*n_coeffs))
    
    # Default case - return all indices
    return list(range(100))  # Large enough for all feature types

def apply_focus_to_feature_vector(features, focus_type, feature_type):
    """
    Apply focus weighting to the feature vector.
    
    Parameters:
    -----------
    features : list or numpy.ndarray
        The feature vector
    focus_type : str
        Type of focus: 'timbre', 'spectral', 'energy', 'temporal', 'full'
    feature_type : str
        The type of MFCC feature
        
    Returns:
    --------
    focused_features : list or numpy.ndarray (same type as input)
        Feature vector with focus weights applied
    """
    if focus_type is None or focus_type == 'full':
        return features
    
    # Convert to numpy array if it's a list
    is_list = isinstance(features, list)
    features_array = np.array(features)
    
    # Get indices for the focus area
    focus_indices = get_feature_vector_focus_indices(feature_type, focus_type)
    
    # Create weight vector (all small values initially)
    weights = np.ones(len(features_array)) * 0.2  # Scale down non-focus areas
    
    # Set weights for focus indices higher
    weights[focus_indices] = 3.0  # Scale up focus areas
    
    # Apply weights
    focused_features = features_array * weights
    
    # Return in the same format as the input
    if is_list:
        return focused_features.tolist()
    return focused_features



# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--force-host', type=bool, default=False, help='Force the host to be the one specified in the host argument.') # e.g for the ROBIN-HPC
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
# parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate of the audio data.')
parser.add_argument('--process-title', type=str, default='features', help='Process title to use.')
parser.add_argument('--host-info-file', type=str, default='', help='Host information file to use.')
parser.add_argument('--models-path', type=str, default='../../measurements/models', help='Path to classification models.')
args = parser.parse_args()

# sample_rate = args.sample_rate

# set PROCESS_TITLE as either the environment variable or the default value
PROCESS_TITLE = os.environ.get('PROCESS_TITLE', args.process_title)
setproctitle(PROCESS_TITLE)

# set PORT as either the environment variable or the default value
PORT = int(os.environ.get('PORT', args.port))

HOST = args.host

MODELS_PATH = args.models_path
# if MODELS_PATH contains "/localscratch/<job-ID>" then replace the job-ID with the environment variable SLURM_JOB_ID
if '/localscratch/' in MODELS_PATH:
    MODELS_PATH = MODELS_PATH.replace('/localscratch/<job-ID>', '/localscratch/' + os.environ.get('SLURM_JOB_ID') )

# if the host-info-file is not empty
if args.host_info_file:
    # automatically assign the host IP from the machine's hostname
    if not args.force_host:
        HOST = os.uname().nodename
    # HOST = socket.gethostname()
    print("HOST", HOST)
    # the host-info-file name ends with "host-" and an index number: host-0, host-1, etc.
    # - for each comonent of that index number, add that number plus 1 to PORT and assign to the variable PORT

    PORT = filepath_to_port(args.host_info_file)

    # write the host IP and port to the host-info-file
    with open(args.host_info_file, 'w') as f:
        f.write('{}:{}'.format(HOST, PORT))


MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB

print('Starting features WebSocket server at ws://{}:{}'.format(HOST, PORT))

# Start the WebSocket server with supplied command line arguments
start_server = websockets.serve(socket_server, 
                                HOST, 
                                PORT,
                                max_size=MAX_MESSAGE_SIZE,
                                ping_timeout=None,
                                ping_interval=None)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()