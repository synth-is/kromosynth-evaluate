# similarity / distance measurements for embeddings / sets of feature vectors

# TODO - just cosine similarity for now, but we could add more metrics such as Euclidean distance, etc.

import asyncio
import websockets
import json
import argparse
import numpy as np
import os
from setproctitle import setproctitle
import time
from urllib.parse import urlparse, parse_qs
from scipy.spatial.distance import cosine

import sys
sys.path.append('../..')
from evaluation.util import filepath_to_port

def str_to_bool(s):
    return s.lower() in ['true', '1', 't', 'y', 'yes']

reference_embeddings = {}
async def socket_server(websocket, path):
    global reference_embeddings
    # Parse the path and query components from the URL
    url_components = urlparse(path)
    request_path = url_components.path  # This holds the path component of the URL
    query_params = parse_qs(url_components.query)  # This will hold the query parameters as a dict
    try:
        # Wait for the first message and determine its type
        message = await websocket.recv()

        start = time.time()
        reference_embedding_path = query_params.get('reference_embedding_path', [None])[0]
        reference_embedding_key = query_params.get('reference_embedding_key', [None])[0]
        if reference_embedding_path is not None and os.path.exists(reference_embedding_path) and reference_embedding_path not in reference_embeddings:
            print(f"Loading reference embeddings from {reference_embedding_path}")
            # reference_embeddings[reference_embedding_path] = np.load(reference_embedding_path, allow_pickle=False).item()
            # read JSON file
            with open(reference_embedding_path, 'r') as f:
                reference_embeddings[reference_embedding_path] = json.load(f)
        query_embedding = json.loads(message)  # receive JSON
        print(f"embeddings shape: {np.array(query_embedding).shape}")
        # if reference_embedding_key contains a comma, it means that we are looking for multiple embeddings, which should be concatenated
        if ',' in reference_embedding_key:
            reference_embedding_keys = reference_embedding_key.split(',')
            reference_embedding = np.concatenate([reference_embeddings[reference_embedding_path][key] for key in reference_embedding_keys], axis=0)
        else:
            reference_embedding = reference_embeddings[reference_embedding_path][reference_embedding_key]
        if request_path == '/cosine':
            cosine_dissimilarity = cosine(query_embedding, reference_embedding)
            fitness = 1 - (cosine_dissimilarity / 2)

            print('fitness:', fitness)

            # the above score is apparently longdouble, and serialising it via JSON is fine on macOS (M1)
            # but errors out in a Linux environment (e.g. Fox HPC), and the following conversion solves that issue:
            # score = np.float64(score)

            response = {'status': 'received standalone audio', 'fitness': fitness}
            await websocket.send(json.dumps(response))

        end = time.time()

        print('Time taken to evaluate fitness (ref_features):', end - start)

    except Exception as e:
        print('quality: Exception', e)
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

print('Starting quality ref features WebSocket server at ws://{}:{}'.format(HOST, PORT))

# Start the WebSocket server with supplied command line arguments
start_server = websockets.serve(socket_server, 
                                HOST, 
                                PORT,
                                max_size=MAX_MESSAGE_SIZE,
                                ping_timeout=None,
                                ping_interval=None)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()