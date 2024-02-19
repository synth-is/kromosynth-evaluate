# 2: websocket server for mood classification audio data
# - https://essentia.upf.edu/models.html 

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
from measurements.quality.quality_mood import mood_aggressive, mood_happy, mood_non_happy, mood_party, mood_relaxed, mood_sad, mood_acoustic, mood_electronic

async def socket_server(websocket, path):
    # Wait for the first message and determine its type
    message = await websocket.recv()

    if isinstance(message, bytes):
        start = time.time()
        # Received binary message (assume it's an audio buffer)
        audio_data = message
        print('Audio data received for fitness evaluation, by sound quality (SQ) metrics')
        # convert the audio data to a numpy array
        audio_data = np.frombuffer(audio_data, dtype=np.float32)
        
        fitness_percentages = []
        for method in QUALITY_METHODS:
          if method == 'mood_aggressive':
            fitness_percentages.append(mood_aggressive(audio_data, MODELS_PATH))
          elif method == 'mood_happy':
            fitness_percentages.append(mood_happy(audio_data, MODELS_PATH))
          elif method == 'mood_non_happy':
            fitness_percentages.append(mood_non_happy(audio_data, MODELS_PATH))
          elif method == 'mood_party':
            fitness_percentages.append(mood_party(audio_data, MODELS_PATH))
          elif method == 'mood_relaxed':
            fitness_percentages.append(mood_relaxed(audio_data, MODELS_PATH))
          elif method == 'mood_sad':
            fitness_percentages.append(mood_sad(audio_data, MODELS_PATH))
          elif method == 'mood_acoustic':
            fitness_percentages.append(mood_acoustic(audio_data, MODELS_PATH))
          elif method == 'mood_electronic':
            fitness_percentages.append(mood_electronic(audio_data, MODELS_PATH))

        print('sound quality percentages (mood):', fitness_percentages)

        # lower value, the better
        fitness_value = sum(fitness_percentages) / len(fitness_percentages)

        print('Fitness value (SQ):', fitness_value)

        end = time.time()
        print('quality_mood: Time taken to evaluate fitness:', end - start)

        response = {'status': 'received standalone audio', 'fitness': fitness_value}
        await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--sample-rate', type=int, default=48000, help='Sample rate of the audio data.')
parser.add_argument('--quality-methods', type=str, default='mood_happy', help='Quality methods to use.')
parser.add_argument('--process-title', type=str, default='quality_mood', help='Process title to use.')
parser.add_argument('--models-path', type=str, default='../../measurements/models', help='Path to classification models.')
args = parser.parse_args()

sample_rate = args.sample_rate

# parse the comma separted list of quality methods
QUALITY_METHODS = args.quality_methods.split(',')

MODELS_PATH = args.models_path

# set PORT as either the environment variable or the default value
PORT = int(os.environ.get('PORT', args.port))

# set PROCESS_TITLE as either the environment variable or the default value
PROCESS_TITLE = os.environ.get('PROCESS_TITLE', args.process_title)
setproctitle(PROCESS_TITLE)

print('Starting fitness / sound quality (mood) WebSocket server at ws://{}:{}'.format(args.host, PORT))
# Start the WebSocket server with supplied command line arguments
start_server = websockets.serve(socket_server, 
                                args.host, 
                                PORT)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()