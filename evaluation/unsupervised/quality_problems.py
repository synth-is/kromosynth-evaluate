# 2: websocket server to determine the fitness / quality / objective value of the audio data
# - based on Audio Quality Control methods: https://www.aes.org/e-lib/browse.cfm?elib=20338

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
from measurements.quality.quality_control import click_count_percentage, discontinuity_count_percentage, gaps_count_percentage, hum_precence_percentage, saturation_percentage, signal_to_noise_percentage_of_excellence, true_peak_clipping_percentage, noise_burst_percentage, compressibility_percentage

async def socket_server(websocket, path):
    # Wait for the first message and determine its type
    message = await websocket.recv()

    if isinstance(message, bytes):
        start = time.time()
        # Received binary message (assume it's an audio buffer)
        audio_data = message
        print('Audio data received for fitness evaluation')
        # convert the audio data to a numpy array
        audio_data = np.frombuffer(audio_data, dtype=np.float32)
        
        fitness_percentages = []
        for method in quality_methods:
            if method == 'click_count_percentage':
              fitness_percentages.append(1 - click_count_percentage(audio_data, sample_rate))
            elif method == 'discontinuity_count_percentage':
              fitness_percentages.append(1 - discontinuity_count_percentage(audio_data, sample_rate))
            elif method == 'gaps_count_percentage':
              fitness_percentages.append(1 - gaps_count_percentage(audio_data, sample_rate))
            elif method == 'hum_precence_percentage':
              fitness_percentages.append(1 - hum_precence_percentage(audio_data, sample_rate))
            elif method == 'saturation_percentage':
              fitness_percentages.append(1 - saturation_percentage(audio_data, sample_rate))
            elif method == 'signal_to_noise_percentage_of_excellence':
              fitness_percentages.append(1 - signal_to_noise_percentage_of_excellence(audio_data, sample_rate))
            elif method == 'true_peak_clipping_percentage':
              fitness_percentages.append(1 - true_peak_clipping_percentage(audio_data, sample_rate))
            elif method == 'noise_burst_percentage':
              fitness_percentages.append(1 - noise_burst_percentage(audio_data, sample_rate))
            elif method == 'compressibility_percentage':
              fitness_percentages.append(compressibility_percentage(audio_data))

        print('Problem percentages:', fitness_percentages)

        # lower value, the better
        fitness_value = sum(fitness_percentages) / len(fitness_percentages)

        end = time.time()
        print('quality_problems: Time taken to evaluate fitness:', end - start)

        print('Fitness value:', fitness_value)

        response = {'status': 'received standalone audio', 'fitness': fitness_value}
        await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate of the audio data.')
parser.add_argument('--quality-methods', type=str, default='click_count_percentage', help='Quality methods to use.')
parser.add_argument('--process-title', type=str, default='quality_problems', help='Process title to use.')
args = parser.parse_args()

sample_rate = args.sample_rate

# parse the comma separted list of quality methods
quality_methods = args.quality_methods.split(',')

# set PORT as either the environment variable or the default value
PORT = int(os.environ.get('PORT', args.port))

# set PROCESS_TITLE as either the environment variable or the default value
PROCESS_TITLE = os.environ.get('PROCESS_TITLE', args.process_title)
setproctitle(PROCESS_TITLE)

print('Starting fitness / quality WebSocket server at ws://{}:{}'.format(args.host, PORT))
# Start the WebSocket server with supplied command line arguments
start_server = websockets.serve(socket_server, 
                                args.host, 
                                PORT)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()