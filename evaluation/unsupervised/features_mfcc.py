# 1: websocket server to extract MFCC features from audio data

import asyncio
import websockets
import json
import argparse
import numpy as np
import os
from setproctitle import setproctitle
import time

# import the function get_mfcc_feature_means_stdv_firstorderdifference_concatenated from measurements/diversity/mfcc.py
import sys
sys.path.append('../..')
from measurements.diversity.audio_features import get_mfcc_feature_means_stdv_firstorderdifference_concatenated

async def socket_server(websocket, path):
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
        features = get_mfcc_feature_means_stdv_firstorderdifference_concatenated(audio_data, sample_rate)
        # print('MFCC features extracted:', features)

        end = time.time()
        print('features_mfcc: Time taken to extract features:', end - start)

        response = {'status': 'received standalone audio', 'features': features.tolist()}
        # response = {'status': 'received standalone audio'}
        await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate of the audio data.')
parser.add_argument('--process-title', type=str, default='features_mfcc', help='Process title to use.')
args = parser.parse_args()

sample_rate = args.sample_rate

# set PORT as either the environment variable or the default value
PORT = int(os.environ.get('PORT', args.port))

# set PROCESS_TITLE as either the environment variable or the default value
PROCESS_TITLE = os.environ.get('PROCESS_TITLE', args.process_title)
setproctitle(PROCESS_TITLE)

MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB

print('Starting features WebSocket server at ws://{}:{}'.format(args.host, PORT))

# Start the WebSocket server with supplied command line arguments
start_server = websockets.serve(socket_server, 
                                args.host, 
                                PORT,
                                max_size=MAX_MESSAGE_SIZE)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()