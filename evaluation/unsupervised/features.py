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
    get_vggish_embeddings, get_vggish_embeddings_essentia,
    get_pann_embeddings, get_pann_embeddings_panns_inference,
    get_clap_embeddings, get_encodec_embeddings,
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
    request_path = url_components.path  # This holds the path component of the URL
    query_params = parse_qs(url_components.query)  # This will hold the query parameters as a dict
    try:
        sample_rate = int(query_params.get('sample_rate', [16000])[0])
        # Wait for the first message and determine its type
        message = await websocket.recv()

        if isinstance(message, bytes):
            start = time.time()
            # Received binary message (assume it's an audio buffer)
            audio_data = message
            print('Audio data received for feature extraction')
            # convert the audio data to a numpy array
            audio_data = np.frombuffer(audio_data, dtype=np.float32)
            # Process the audio data...

            if request_path == '/mfcc' or request_path == '/':
                print('Extracting MFCC features...')
                embeddings = get_mfcc_features(audio_data, sample_rate)
                features = get_feature_means_stdv_firstorderdifference_concatenated(embeddings)
                features_type = 'mfcc'
            elif request_path == '/mfcc-sans0': # without the first coefficient
                print('Extracting MFCC features without the first coefficient...')
                embeddings = get_mfcc_features(audio_data, sample_rate)
                embeddings = embeddings[1:]
                features = get_feature_means_stdv_firstorderdifference_concatenated(embeddings)
                features_type = 'mfcc-sans0'
            elif request_path == '/mfcc-statistics':
                print('Extracting MFCC statistics...')
                embeddings = get_mfcc_features(audio_data, sample_rate)
                features = compute_feature_statistics(embeddings)
                features_type = 'mfcc-statistics'
            elif request_path == '/mfcc-sans0-statistics':
                print('Extracting MFCC statistics without the first coefficient...')
                embeddings = get_mfcc_features(audio_data, sample_rate)
                embeddings = embeddings[1:]
                features = compute_feature_statistics(embeddings)
                features_type = 'mfcc-sans0-statistics'
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