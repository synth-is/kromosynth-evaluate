# similarity / distance measurements for audio buffers

# TODO https://github.com/pranaymanocha/PerceptualAudio/tree/master/cdpam

# TODO https://github.com/RiccardoVib/Physics-Informed-Differentiable-Piano/blob/main/Code/NFLossFunctions.py

import asyncio
import websockets
import json
import argparse
import numpy as np
import os
from setproctitle import setproctitle
import time
from urllib.parse import urlparse, parse_qs
import sys
sys.path.append('../..')
from measurements.quality.quality_ref_buffer import get_cdpam_distance, get_multi_resolution_spectral_loss
from util import filepath_to_port
import cdpam

def str_to_bool(s):
    return s.lower() in ['true', '1', 't', 'y', 'yes']

# TODO: this is incomplete and has issues: see comment in measurements/quality/quality_ref_buffer.py

async def socket_server(websocket, path):
    # Parse the path and query components from the URL
    url_components = urlparse(path)
    request_path = url_components.path  # This holds the path component of the URL
    query_params = parse_qs(url_components.query)  # This will hold the query parameters as a dict
    try:
        # Wait for the first message and determine its type
        message = await websocket.recv()

        if isinstance(message, bytes):
          print("Received binary message (assume it's an audio buffer)")
          query_audio = np.frombuffer(message, dtype=np.float32)
          start = time.time()
          reference_audio_path = query_params.get('reference_audio_path', [None])[0]
          if request_path == '/cdpam':
            print("Calculating CDPAM distance for embedding")
            fitness = get_cdpam_distance(query_audio, reference_audio_path)

            response = {'status': 'received standalone audio', 'fitness': fitness}
            await websocket.send(json.dumps(response))
          elif request_path == '/stft-loss':
            print("Calculating cosine distance for embedding")
            fitness = get_multi_resolution_spectral_loss(query_audio, reference_audio_path)

            response = {'status': 'received standalone audio', 'fitness': fitness}
            await websocket.send(json.dumps(response))

          end = time.time()

          print('Time taken to evaluate fitness (FAD):', end - start)
        else:
          print("Received JSON message with two audio file paths")
          message = json.loads(message)
          start = time.time()
          file1 = message['file1']
          file2 = message['file2']
          if request_path == '/cdpam':
            query_audio = cdpam.load_audio(file1)
            print("Calculating CDPAM distance between two audio file paths")
            fitness = get_cdpam_distance(query_audio, file2)
            print('Fitness value (CDPAM):', fitness)
            response = {'status': 'received standalone audio', 'fitness': fitness}
            await websocket.send(json.dumps(response))
          elif request_path == '/stft-loss':
            query_audio = np.frombuffer(open(file1, 'rb').read(), dtype=np.float32)
            print("Calculating cosine distance between two audio file paths")
            fitness = get_multi_resolution_spectral_loss(query_audio, file2)
            print('Fitness value (STFT loss):', fitness)
            response = {'status': 'received standalone audio', 'fitness': fitness}
            await websocket.send(json.dumps(response))

    except Exception as e:
        print('quality_ref_buffer: Exception', e)
        response = {'status': 'ERROR', 'error': str(e) }
        await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--force-host', type=bool, default=False, help='Force the host to be the one specified in the host argument.') # e.g for the ROBIN-HPC
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--process-title', type=str, default='quality_fad', help='Process title to use.')
parser.add_argument('--host-info-file', type=str, default='', help='Host information file to use.')
args = parser.parse_args()

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

print('Starting quality_ref_buffer WebSocket server at ws://{}:{}'.format(HOST, PORT))

# Start the WebSocket server with supplied command line arguments
start_server = websockets.serve(socket_server, 
                                HOST, 
                                PORT,
                                max_size=MAX_MESSAGE_SIZE,
                                ping_timeout=None,
                                ping_interval=None)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()