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
    get_mfcc_features, get_vggish_embeddings, get_pann_embeddings, get_clap_embeddings, get_encodec_embeddings
)
from util import filepath_to_port

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
            else:
                ckpt_dir = query_params.get('ckpt_dir', [None])[0]
                audio_data = [audio_data]
                print('ckpt_dir:', ckpt_dir)
                if request_path == '/vggish':
                    print('Extracting VGGish embeddings...')
                    use_pca = query_params.get('use_pca', [False])[0]
                    use_activation = query_params.get('use_activation', [False])[0]
                    embeddings = get_vggish_embeddings(audio_data, sample_rate, ckpt_dir, use_pca, use_activation)
                    features = get_feature_means_stdv_firstorderdifference_concatenated(embeddings.T) # transpose the embeddings to match the shape of the MFCC features: (num_features, num_frames
                elif request_path == '/pann':
                    print('Extracting PANN embeddings...')
                    embeddings = get_pann_embeddings(audio_data, sample_rate, ckpt_dir)
                    features = embeddings
                elif request_path == '/clap':
                    print('Extracting CLAP embeddings...')
                    submodel_name = query_params.get('submodel_name', ["630k-audioset"])[0]
                    enable_fusion = query_params.get('enable_fusion', [False])[0]
                    embeddings = get_clap_embeddings(audio_data, sample_rate, ckpt_dir, submodel_name, enable_fusion)
                    features = embeddings[0]
                elif request_path == '/encodec':
                    print('Extracting EnCodec embeddings...')
                    embeddings = get_encodec_embeddings(audio_data, sample_rate, ckpt_dir)
                    features = get_feature_means_stdv_firstorderdifference_concatenated(embeddings)

            print('Embeddings extracted shape:', embeddings.shape)

            # features = get_feature_means_stdv_firstorderdifference_concatenated(embeddings)

            print('Features extracted shape:', features.shape)

            # print('MFCC features extracted:', features)

            end = time.time()
            print('features_mfcc: Time taken to extract features:', end - start)

            response = {'status': 'received standalone audio', 'features': features.tolist(), 'embedding': embeddings.tolist()}
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
args = parser.parse_args()

# sample_rate = args.sample_rate

# set PROCESS_TITLE as either the environment variable or the default value
PROCESS_TITLE = os.environ.get('PROCESS_TITLE', args.process_title)
setproctitle(PROCESS_TITLE)

# set PORT as either the environment variable or the default value
PORT = int(os.environ.get('PORT', args.port))

HOST = args.host

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