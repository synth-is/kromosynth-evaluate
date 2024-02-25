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
from measurements.quality.quality_instrumentation import nsynth_instrument_mean
from util import filepath_to_port

async def socket_server(websocket, path):
    try:
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
            if method == 'nsynth_instrument':
              fitness_percentages.append(nsynth_instrument_mean(audio_data, MODELS_PATH))

          # lower value, the better
          # fitness_value = sum(fitness_percentages) / len(fitness_percentages)
          fitness_value = fitness_percentages[0]

          print('Fitness value (instrumentation):', fitness_value)

          end = time.time()
          print('nsynth_instrument: Time taken to evaluate fitness:', end - start)

          response = {'status': 'received standalone audio', 'fitness': json.dumps(fitness_value.item())} # https://stackoverflow.com/questions/53082708/typeerror-object-of-type-float32-is-not-json-serializable#comment93577758_53082860
          await websocket.send(json.dumps(response))
    except Exception as e:
        print('nsynth_instrument: Exception', e)
        response = {'status': 'ERROR'}
        await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--force-host', type=bool, default=False, help='Force the host to be the one specified in the host argument.') # e.g for the ROBIN-HPC
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--sample-rate', type=int, default=48000, help='Sample rate of the audio data.')
parser.add_argument('--quality-methods', type=str, default='nsynth_instrument', help='Quality methods to use.')
parser.add_argument('--process-title', type=str, default='quality_mood', help='Process title to use.')
parser.add_argument('--models-path', type=str, default='../../measurements/models', help='Path to classification models.')
parser.add_argument('--host-info-file', type=str, default='', help='Host information file to use.')
args = parser.parse_args()

sample_rate = args.sample_rate

# parse the comma separted list of quality methods
print("args.quality_methods", args.quality_methods)
QUALITY_METHODS = args.quality_methods.split(',')

MODELS_PATH = args.models_path

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
    # the host-info-file name ends with "host-" and an index number: host-0, host-1, etc.
    # - for each comonent of that index number, add that number plus 1 to PORT and assign to the variable PORT

    PORT = filepath_to_port(args.host_info_file)

    # write the host IP and port to the host-info-file
    with open(args.host_info_file, 'w') as f:
        f.write('{}:{}'.format(HOST, PORT))

MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB

print('Starting fitness / sound quality (mood) WebSocket server at ws://{}:{}'.format(HOST, PORT))
# Start the WebSocket server with supplied command line arguments
start_server = websockets.serve(socket_server, 
                                HOST, 
                                PORT,
                                max_size=MAX_MESSAGE_SIZE,
                                ping_timeout=None,
                                ping_interval=None)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()