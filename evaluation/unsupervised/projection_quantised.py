# 3: to determine the diversity of a set of solutions, project the feature vectors onto a 2D plane using PCA, then quantise the projection

import asyncio
import websockets
import json
import argparse
import os
from setproctitle import setproctitle
import time
from urllib.parse import urlparse, parse_qs
import numpy as np


import sys
sys.path.append('../..')
from measurements.diversity.dimensionality_reduction import get_pca_projection, get_autoencoder_projection, get_umap_projection, clear_tf_session, projection_with_cleanup
from measurements.diversity.util.discretise import vector_to_index
# from measurements.diversity.util.metrics_visualiser import MetricsVisualizer
from measurements.diversity.util.diversity_metrics import calculate_diversity_metrics, perform_cluster_analysis, calculate_performance_spread, calculate_novelty_metric
from evaluation.util import filepath_to_port

def remove_duplicates_keep_highest(discretised_projection, fitness_values):
    if len(discretised_projection) != len(fitness_values):
        raise ValueError("The lengths of discretised_projection and fitness_values must match.")

    # Create a dictionary to keep the highest fitness value for each unique tuple
    best_fit_dict = {}
    for idx, tup in enumerate(discretised_projection):
        print('idx', idx, 'tup', tup)
        if tup not in best_fit_dict or best_fit_dict[tup]['score'] < fitness_values[idx]:
            best_fit_dict[tup] = {'index': idx, 'score': fitness_values[idx]}
    
    print('best_fit_dict', best_fit_dict)

    # Extract the indices of the tuples with the highest fitness values, ordered by index
    # indices_to_keep = [info['index'] for tup, info in best_fit_dict.items()]
    # print('indices_to_keep', indices_to_keep)
    indices_to_keep_sorted = [info['index'] for tup, info in sorted(best_fit_dict.items(), key=lambda item: item[1]['index'])]
    print('indices_to_keep_sorted', indices_to_keep_sorted)

    # Use the indices to construct the arrays of unique tuples and corresponding fitness values
    unique_projection = [discretised_projection[idx] for idx in indices_to_keep_sorted]
    unique_fitness_values = [fitness_values[idx] for idx in indices_to_keep_sorted]

    return unique_projection, unique_fitness_values, indices_to_keep_sorted

cell_range_min_for_projection = {}
cell_range_max_for_projection = {}
async def socket_server(websocket, path):
    global cell_range_min_for_projection
    global cell_range_max_for_projection
    # start time
    start = time.time()
    try:
        data = await websocket.recv()

        jsonData = json.loads(data)  # receive JSON

        feature_vectors = jsonData.get('feature_vectors', [])
        should_fit = jsonData.get('should_fit', False)
        print('should_fit: ', should_fit)
        calculate_surprise = jsonData.get('calculate_surprise', False)
        use_autoencoder_for_surprise = jsonData.get('use_autoencoder_for_surprise', False)
        calculate_novelty = jsonData.get('calculate_novelty', False)
        print('calculate_surprise: ', calculate_surprise)
        evorun_dir = jsonData.get('evorun_dir', '')

        url_components = urlparse(path)
        request_path = url_components.path

        if request_path == '/pca':
            pca_components = jsonData.get('pca_components')
            # parse the comma separated string of dimensions into a list of integers, if it is not empty
            if pca_components is not None and pca_components != '':
                components_list = list(map(int, pca_components.split(',')))
            else:
                components_list = []
            projection, surprise_scores = projection_with_cleanup(
                get_pca_projection,
                feature_vectors, dimensions, should_fit, evorun_dir, calculate_surprise, components_list, use_autoencoder_for_surprise
            )
            # get_pca_projection(feature_vectors, dimensions, should_fit, evorun_dir, calculate_surprise, components_list, use_autoencoder_for_surprise)
            # if use_autoencoder_for_surprise:
                # clear_tf_session()
        elif request_path == '/autoencoder':
            projection, surprise_scores = projection_with_cleanup(
                get_autoencoder_projection,
                feature_vectors, dimensions, should_fit, evorun_dir, calculate_surprise
            )
            # get_autoencoder_projection(feature_vectors, dimensions, should_fit, evorun_dir, calculate_surprise)
            # clear_tf_session()
        elif request_path == '/umap':
            projection, surprise_scores = projection_with_cleanup(
                get_umap_projection,
                feature_vectors, dimensions, should_fit, evorun_dir, calculate_surprise,
                metric='cosine'
            )
            #get_umap_projection(feature_vectors, dimensions, should_fit, evorun_dir, calculate_surprise, 
            #                                                 metric='cosine')
            # clear_tf_session()
        elif request_path == '/raw':
            projection = np.array(feature_vectors)
            surprise_scores = None


        elif request_path == '/diversity_metrics':
            generation = jsonData.get('generation', 0)
            feature_vectors = jsonData['feature_vectors']
            stage = jsonData.get('stage', '')  # 'before' or 'after'
            
            diversity_metrics = calculate_diversity_metrics(feature_vectors)
            response = {
                'status': 'OK', 
                'diversity_metrics': diversity_metrics,
                'generation': generation,
                'stage': stage
            }

        elif request_path == '/cluster_analysis':
            generation = jsonData.get('generation', 0)
            stage = jsonData.get('stage', '')  # 'before' or 'after'
            feature_vectors = jsonData['feature_vectors']
            cluster_analysis = perform_cluster_analysis(feature_vectors)
            response = {
                'status': 'OK', 'cluster_analysis': cluster_analysis,
                'generation': generation,
                'stage': stage
                }

        elif request_path == '/performance_spread':
            generation = jsonData.get('generation', 0)
            stage = jsonData.get('stage', '')  # 'before' or 'after'
            feature_vectors = jsonData['feature_vectors']
            fitness_values = jsonData['fitness_values']
            performance_spread = calculate_performance_spread(feature_vectors, fitness_values)
            response = {
                'status': 'OK', 'performance_spread': performance_spread,
                'generation': generation,
                'stage': stage
                }

        # elif request_path == '/visualize_metrics':
        #     metrics_history = jsonData.get('metrics_history', {})
        #     diversity_dir = jsonData['diversity_dir']
        #     visualizer = MetricsVisualizer(diversity_dir + '/viz')
        #     visualizer.visualize(metrics_history)
        #     response = {'status': 'OK', 'message': 'Visualizations created successfully'}

        if should_fit:
            # delete the entry for the request_path in cell_range_min_for_projection and cell_range_max_for_projection
            if request_path in cell_range_min_for_projection:
                del cell_range_min_for_projection[request_path]
            if request_path in cell_range_max_for_projection:
                del cell_range_max_for_projection[request_path]


        # if projection is defined
        if 'projection' in locals():
            print('projection', projection)

            projection_min = projection.min()
            projection_max = projection.max()
            if request_path not in cell_range_min_for_projection or cell_range_min_for_projection[request_path] > projection_min:
                cell_range_min_for_projection[request_path] = projection_min
            if request_path not in cell_range_max_for_projection or cell_range_max_for_projection[request_path] < projection_max:
                cell_range_max_for_projection[request_path] = projection_max
        

            # taking min/max of the projection values doesn't work, when there is just one (or few) vectors incoming - fixing to the range 0 to 1 for now:
            # cell_range_min = projection.min()
            # cell_range_max = projection.max()
            # cell_range_min = 0
            # cell_range_max = 1
            cell_range_min = cell_range_min_for_projection[request_path]
            cell_range_max = cell_range_max_for_projection[request_path]
            print('cell_range_min', cell_range_min)
            print('cell_range_max', cell_range_max)
            # discretise / quantise the projection
            # for each element in the projection, calculate the index of the cell it belongs to
            # scale the range of each cell value to the range 0 to 1 with a call like:
            # vector_to_index(v, cell_range_min, cell_range_max, cells, dimensions)
            cells = args.dimension_cells
            discretised_projection = []
            for element in projection:
                discretised_vector = vector_to_index(element, cell_range_min, cell_range_max, cells, dimensions)
                print('discretised_vector', discretised_vector)
                discretised_projection.append(discretised_vector)

            print('discretised_projection size: ', len(discretised_projection))

            if calculate_novelty:
                novelty_scores = calculate_novelty_metric(discretised_projection, k=15)
            else:
                novelty_scores = None

        # if 'response' not in locals(): # endpoints /pca, /umap, /raw - TODO clean up
            response = {
                'status': 'OK', 
                'feature_map': discretised_projection, 
                'surprise_scores': surprise_scores.tolist() if surprise_scores is not None else None,
                'novelty_scores': novelty_scores.tolist() if novelty_scores is not None else None
                }
        await websocket.send(json.dumps(response))
        
    except Exception as e:
        print('Error in projection_quantised.py:', e)
        response = {'status': 'ERROR' + str(e)}
        await websocket.send(json.dumps(response))
        return

    end = time.time()
    print('projection_quantised: Time taken to process: ', end - start)

    

    # TOD abandoning filtering of duplicates for now, as elite replacement currently handles this, in quality-diversity-search.js (kromosynth-qd)
    # ensure there is only one element per cell
    # unique_projection, unique_fitness_values, indices_to_keep = remove_duplicates_keep_highest(discretised_projection, fitness_values)

    # print(unique_projection)  # Should print the unique tuples
    # print(unique_fitness_values)  # Should print corresponding highest fitness values

    # Send a JSON response back to the client
    # response = {'status': 'OK', 'feature_map': unique_projection, 'fitness_values': unique_fitness_values, 'indices_to_keep': indices_to_keep}


# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--force-host', type=bool, default=False, help='Force the host to be the one specified in the host argument.') # e.g for the ROBIN-HPC
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--dimensions', type=int, default=2, help='Number of dimensions to reduce to.')
parser.add_argument('--dimension-cells', type=int, default=10, help='Number of cells in each dimension.')
parser.add_argument('--process-title', type=str, default='projection_quantised', help='Process title to use.')
parser.add_argument('--host-info-file', type=str, default='', help='Host information file to use.')
parser.add_argument('--plot-variance-ratio', type=bool, default=False, help='Plot the variance ratio of the PCA model.')
args = parser.parse_args()

dimensions = args.dimensions
# plot_variance_ratio = args.plot_variance_ratio

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

print('Starting projection WebSocket server at ws://{}:{}'.format(HOST, PORT))
start_server = websockets.serve(socket_server, 
                                HOST, 
                                PORT,
                                max_size=MAX_MESSAGE_SIZE,
                                ping_timeout=None,
                                ping_interval=None)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()