# 3: to determine the diversity of a set of solutions, project the feature vectors onto a 2D plane using PCA, then quantise the projection

import asyncio
import websockets
import json
import argparse

import sys
sys.path.append('../..')
from measurements.diversity.dimensionality_reduction import get_pca_projection
from measurements.diversity.util.discretise import vector_to_index

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

async def socket_server(websocket, path):
    data = await websocket.recv()
    jsonData = json.loads(data)  # receive JSON

    # print('JSON data received: ', jsonData)

    try:
        projection = get_pca_projection(jsonData['feature_vectors'], dimensions)
    except ValueError as e:
        response = {'status': 'ERROR', 'message': str(e)}
        await websocket.send(json.dumps(response))
        return

    # print('projection: ', projection)

    # fitness_values = jsonData['fitness_values']

    cell_range_min = 0
    cell_range_max = 1
    # discretise / quantise the projection
    # for each element in the projection, calculate the index of the cell it belongs to
    # scale the range of each cell value to the range 0 to 1 with a call like:
    # vector_to_index(v, cell_range_min, cell_range_max, cells, dimensions)
    cells = args.dimension_cells
    discretised_projection = []
    for element in projection:
        discretised_projection.append(vector_to_index(element, cell_range_min, cell_range_max, cells, dimensions))

    print('discretised_projection size: ', len(discretised_projection))

    # TOD abandoning filtering of duplicates for now, as elite replacement currently handles this, in quality-diversity-search.js (kromosynth-qd)
    # ensure there is only one element per cell
    # unique_projection, unique_fitness_values, indices_to_keep = remove_duplicates_keep_highest(discretised_projection, fitness_values)

    # print(unique_projection)  # Should print the unique tuples
    # print(unique_fitness_values)  # Should print corresponding highest fitness values

    # Send a JSON response back to the client
    # response = {'status': 'OK', 'feature_map': unique_projection, 'fitness_values': unique_fitness_values, 'indices_to_keep': indices_to_keep}
    response = {'status': 'OK', 'feature_map': discretised_projection}
    await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--dimensions', type=int, default=2, help='Number of dimensions to reduce to.')
parser.add_argument('--dimension-cells', type=int, default=10, help='Number of cells in each dimension.')
args = parser.parse_args()

print('Starting projection WebSocket server at ws://{}:{}'.format(args.host, args.port))

dimensions = args.dimensions

MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB
start_server = websockets.serve(socket_server, 
                                args.host, 
                                args.port,
                                max_size=MAX_MESSAGE_SIZE)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()