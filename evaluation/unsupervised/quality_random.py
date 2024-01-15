# 2: websocket server to determine the fitness / quality / objective value of the audio data

import asyncio
import websockets
import json
import argparse
import numpy as np

# import the function get_mfcc_feature_means_stdv_firstorderdifference_concatenated from measurements/diversity/mfcc.py
import sys
sys.path.append('../..')
from measurements.quality.performance import fitness_random

async def socket_server(websocket, path):
    # Wait for the first message and determine its type
    message = await websocket.recv()

    if isinstance(message, bytes):
        # Received binary message (assume it's an audio buffer)
        audio_data = message
        print('Audio data received for fitness evaluation')
        # convert the audio data to a numpy array
        audio_data = np.frombuffer(audio_data, dtype=np.float32)
        
        # Process the audio data...
        fitness_value = fitness_random(audio_data, sample_rate)
        # print('Fitness value:', fitness_value)

        response = {'status': 'received standalone audio', 'fitness': fitness_value}
        await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate of the audio data.')
args = parser.parse_args()

sample_rate = args.sample_rate

print('Starting fitness / quality WebSocket server at ws://{}:{}'.format(args.host, args.port))

# Start the WebSocket server with supplied command line arguments
start_server = websockets.serve(socket_server, args.host, args.port)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()