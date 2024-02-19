# 2: websocket server to determine the fitness / quality / objective value of the audio data
# - based on based on Sound quality (SQ) metrics: https://github.com/Eomys/MoSQITo

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
from measurements.quality.quality_psychoacoustic import (
  roughness_dw_score_average, roughness_dw_score_median, roughness_dw_score_95th_percentile,
  loudness_zwicker_score_average, loudness_zwicker_score_median, loudness_zwicker_score_95th_percentile,
  equal_loudness_contour_score_average, equal_loudness_contour_score_median, equal_loudness_contour_score_95th_percentile
)

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
        for method in quality_methods:
          if method == 'roughness_average':
            fitness_percentages.append(1 - roughness_dw_score_average(audio_data, sample_rate))
          elif method == 'roughness_median':
            fitness_percentages.append(1 - roughness_dw_score_median(audio_data, sample_rate))
          elif method == 'roughness_95th_percentile':
            fitness_percentages.append(1 - roughness_dw_score_95th_percentile(audio_data, sample_rate))
          elif method == 'loudness_average':
            fitness_percentages.append(loudness_zwicker_score_average(audio_data, sample_rate))
          elif method == 'loudness_median':
            fitness_percentages.append(loudness_zwicker_score_median(audio_data, sample_rate))
          elif method == 'loudness_95th_percentile':
            fitness_percentages.append(loudness_zwicker_score_95th_percentile(audio_data, sample_rate))
          elif method == 'equal_loudness_contour_average':
            fitness_percentages.append(equal_loudness_contour_score_average(audio_data, sample_rate))
          elif method == 'equal_loudness_contour_median':
            fitness_percentages.append(equal_loudness_contour_score_median(audio_data, sample_rate))
          elif method == 'equal_loudness_contour_95th_percentile':
            fitness_percentages.append(equal_loudness_contour_score_95th_percentile(audio_data, sample_rate))

        print('sound quality percentages (psychoacoustic):', fitness_percentages)

        # lower value, the better
        fitness_value = sum(fitness_percentages) / len(fitness_percentages)

        print('Fitness value (SQ):', fitness_value)

        end = time.time()
        print('quality_psychoacoustic: Time taken to evaluate fitness:', end - start)

        response = {'status': 'received standalone audio', 'fitness': fitness_value}
        await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--sample-rate', type=int, default=48000, help='Sample rate of the audio data.')
parser.add_argument('--quality-methods', type=str, default='roughness_median', help='Quality methods to use.')
parser.add_argument('--process-title', type=str, default='quality_psychoacoustic', help='Process title to use.')
parser.add_argument('--host-info-file', type=str, default='', help='Host information file to use.')
args = parser.parse_args()

sample_rate = args.sample_rate

# parse the comma separted list of quality methods
quality_methods = args.quality_methods.split(',')

# set PROCESS_TITLE as either the environment variable or the default value
PROCESS_TITLE = os.environ.get('PROCESS_TITLE', args.process_title)
setproctitle(PROCESS_TITLE)

# set PORT as either the environment variable or the default value
PORT = int(os.environ.get('PORT', args.port))

HOST = args.host

# if the host-info-file is not empty
if args.host_info_file:
    # automatically assign the host IP from the machine's hostname
    HOST = os.uname().nodename
    # the host-info-file name ends with "host-" and an index number: host-0, host-1, etc.
    # - for each comonent of that index number, add that number plus 1 to PORT and assign to the variable PORT

    # set host_info_file_index as the index after "host-" in the host-info-file
    host_info_file_index = args.host_info_file.split('host-')[1]
    # add that index to PORT
    PORT += int(host_info_file_index) + 1

    # write the host IP and port to the host-info-file
    with open(args.host_info_file, 'w') as f:
        f.write('{}:{}'.format(HOST, PORT))

MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB

print('Starting fitness / sound quality (psychoacoustic) WebSocket server at ws://{}:{}'.format(HOST, PORT))
# Start the WebSocket server with supplied command line arguments
start_server = websockets.serve(socket_server, 
                                HOST, 
                                PORT,
                                max_size=MAX_MESSAGE_SIZE)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()