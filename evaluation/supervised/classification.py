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
from measurements.quality.quality_instrumentation import (
  nsynth_tagged_predictions, yamnet_tagged_predictions, mtg_jamendo_instrument_predictions,
  music_loop_instrument_role_predictions, mood_acoustic_predictions, mood_electronic_predictions, voice_instrumental_predictions, voice_gender_predictions,
  timbre_predictions, nsynth_acoustic_electronic_predictions, nsynth_bright_dark_predictions, nsynth_reverb_predictions
)
from evaluation.util import filepath_to_port

async def socket_server(websocket, path):
    try:
      # Wait for the first message and determine its type
      message = await websocket.recv()

      if isinstance(message, bytes):
          start = time.time()
          # Received binary message (assume it's an audio buffer)
          audio_data = message
          print('Audio data received for fitness evaluation, by instrumentation metrics')
          # convert the audio data to a numpy array
          audio_data = np.frombuffer(audio_data, dtype=np.float32)

          url_components = urlparse(path)
          query_params = parse_qs(url_components.query)  # This will hold the query parameters as a dict
          classifiers = query_params.get('classifiers', ['nsynth'])[0]
          classifiers_list = classifiers.split(',')
          
          tagged_predictions = {}
          for method in classifiers_list:
            # instruments
            if method == 'nsynth':
              nsynth_dict = nsynth_tagged_predictions(audio_data, MODELS_PATH)
              # merge with the tagged_predictions dictionary
              tagged_predictions = {**tagged_predictions, **nsynth_dict}
            elif method == 'mtg_jamendo_instrument':
              mtg_jamendo_dict = mtg_jamendo_instrument_predictions(audio_data, MODELS_PATH)
              tagged_predictions = {**tagged_predictions, **mtg_jamendo_dict}

            # instrumentation
            elif method == 'music_loop_instrument_role':
              music_loop_dict = music_loop_instrument_role_predictions(audio_data, MODELS_PATH)
              tagged_predictions = {**tagged_predictions, **music_loop_dict}
            elif method == 'mood_acoustic':
              mood_acoustic_dict = mood_acoustic_predictions(audio_data, MODELS_PATH)
              tagged_predictions = {**tagged_predictions, **mood_acoustic_dict}
            elif method == 'mood_electronic':
              mood_electronic_dict = mood_electronic_predictions(audio_data, MODELS_PATH)
              tagged_predictions = {**tagged_predictions, **mood_electronic_dict}
            elif method == 'voice_instrumental':
              voice_instrumental_dict = voice_instrumental_predictions(audio_data, MODELS_PATH)
              tagged_predictions = {**tagged_predictions, **voice_instrumental_dict}
            elif method == 'voice_gender':
              voice_gender_dict = voice_gender_predictions(audio_data, MODELS_PATH)
              tagged_predictions = {**tagged_predictions, **voice_gender_dict}
            elif method == 'timbre':
              timbre_dict = timbre_predictions(audio_data, MODELS_PATH)
              tagged_predictions = {**tagged_predictions, **timbre_dict}
            elif method == 'nsynth_acoustic_electronic':
              nsynth_acoustic_electronic_dict = nsynth_acoustic_electronic_predictions(audio_data, MODELS_PATH)
              tagged_predictions = {**tagged_predictions, **nsynth_acoustic_electronic_dict}
            elif method == 'nsynth_bright_dark':
              nsynth_bright_dark_dict = nsynth_bright_dark_predictions(audio_data, MODELS_PATH)
              tagged_predictions = {**tagged_predictions, **nsynth_bright_dark_dict}
            elif method == 'nsynth_reverb':
              nsynth_reverb_dict = nsynth_reverb_predictions(audio_data, MODELS_PATH)
              tagged_predictions = {**tagged_predictions, **nsynth_reverb_dict}

            # audio events
            elif method == 'yamnet':
              yamnet_dict = yamnet_tagged_predictions(audio_data, MODELS_PATH)
              tagged_predictions = {**tagged_predictions, **yamnet_dict}

          end = time.time()
          print('classification: Time taken to evaluate fitness:', end - start)

          response = {'status': 'received standalone audio', 'taggedPredictions': tagged_predictions} # https://stackoverflow.com/questions/53082708/typeerror-object-of-type-float32-is-not-json-serializable#comment93577758_53082860
          await websocket.send(json.dumps(response))
    except Exception as e:
        print('nsynth_instrument: Exception', e)
        response = {'status': 'ERROR' + str(e)}
        await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--force-host', type=bool, default=False, help='Force the host to be the one specified in the host argument.') # e.g for the ROBIN-HPC
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--sample-rate', type=int, default=48000, help='Sample rate of the audio data.')
parser.add_argument('--process-title', type=str, default='classification', help='Process title to use.')
parser.add_argument('--models-path', type=str, default='../../measurements/models', help='Path to classification models.')
parser.add_argument('--host-info-file', type=str, default='', help='Host information file to use.')
args = parser.parse_args()

sample_rate = args.sample_rate

MODELS_PATH = args.models_path
# if MODELS_PATH contains "/localscratch/<job-ID>" then replace the job-ID with the environment variable SLURM_JOB_ID
if '/localscratch/' in MODELS_PATH:
    MODELS_PATH = MODELS_PATH.replace('/localscratch/<job-ID>', '/localscratch/' + os.environ.get('SLURM_JOB_ID') )

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

print('Starting classification WebSocket server at ws://{}:{}'.format(HOST, PORT))
# Start the WebSocket server with supplied command line arguments
start_server = websockets.serve(socket_server, 
                                HOST, 
                                PORT,
                                max_size=MAX_MESSAGE_SIZE,
                                ping_timeout=None,
                                ping_interval=None)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()