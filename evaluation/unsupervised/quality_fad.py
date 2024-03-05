# a websocket server that has two endpoints:
# - /score
# - - accepting:
# - - embeddings for audio files
# - - path to a reference set embeddings
# - - optional path to a query set embeddings 
#     (which will be combined with the single audio file embedding)

# - /add-to-query-embeddings
# - - accepting:
# - - embedding for a single audio file
# - - (genome) ID of embedding to add
# - - optional (genome) ID of embedding to remove (replaced by the new embedding)
# - - path to a query set embeddings  
#     (which will be combined with the single audio file embedding and saved to the query set embeddings)

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
from measurements.quality.quality_fad import get_fad_score
from util import filepath_to_port

async def socket_server(websocket, path):
    # Parse the path and query components from the URL
    url_components = urlparse(path)
    request_path = url_components.path  # This holds the path component of the URL
    query_params = parse_qs(url_components.query)  # This will hold the query parameters as a dict
    try:
        # Wait for the first message and determine its type
        message = await websocket.recv()

        start = time.time()

        if request_path == '/score':
            print("Calculating FAD score for embedding (potentially combined with population embeddings)")
            embedding = json.loads(message)  # receive JSON
            background_embds_path = query_params.get('background_embds_path', [None])[0]
            eval_embds_path = query_params.get('eval_embds_path', [None])[0]
            measure_collective_performance = query_params.get('measure_collective_performance', [False])[0]
            ckpt_dir = query_params.get('ckpt_dir', [None])[0]

            embeddings = []
            if measure_collective_performance:
                population_embds = np.load(eval_embds_path)
                # population_embeds is a dictionary, with keys being the genome IDs and values being the embeddings
                for genome_id, embedding in population_embds.items():
                    embeddings.append(embedding)
                # embeddings = list(population_embds.values())
            embeddings.append(embedding)

            score = get_fad_score(embeddings, background_embds_path, ckpt_dir)

            response = {'status': 'received standalone audio', 'fitness': score}
            await websocket.send(json.dumps(response))

        elif request_path == '/add-to-query-embeddings':
            print("Adding embedding to query embeddings")
            json_data = json.loads(message)  # receive JSON
            embedding = json_data['embedding']
            candidate_id = json_data['candidate_id']
            replacement_id = json_data.get('replacement_id', None)
            eval_embds_path = query_params.get('eval_embds_path', [None])[0]
            population_embds = np.load(eval_embds_path)
            population_embds[candidate_id] = embedding
            if replacement_id:
                del population_embds[replacement_id]
            np.save(eval_embds_path, population_embds)

            response = {'status': 'OK'}
            await websocket.send(json.dumps(response))

        end = time.time()

        print('Time taken to evaluate fitness (FAD):', end - start)

    except Exception as e:
        print('features: Exception', e)
        response = {'status': 'ERROR'}
        await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--force-host', type=bool, default=False, help='Force the host to be the one specified in the host argument.') # e.g for the ROBIN-HPC
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--process-title', type=str, default='features_mfcc', help='Process title to use.')
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