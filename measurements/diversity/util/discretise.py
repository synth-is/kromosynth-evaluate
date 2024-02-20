def vector_to_index(v, range_min, range_max, cells, dimensions):
    # print('v', v, 'range_min', range_min, 'range_max', range_max, 'cells', cells, 'dimensions', dimensions)
    
    bounds = [(range_min, range_max, cells)] * dimensions
    # print('bounds', bounds)
    n = dimensions

    # Calculate the discrete index for each dimension 
    index = [0]*n

    for i in range(n):
        low, high, cells = bounds[i]
        # print('low', low, 'high', high, 'cells', cells, 'v[i]', v[i])

        # if int(v[i]) < low or int(v[i]) > high: # int cast to avoid float comparison errors, e.g. v[i] can sometimes be 1.0000000000000002 instead of 1.0
        #     raise ValueError(f"Value v[{i}] is out of bounds.")
        
        # if v[i] < low or v[i] > high, assume it is falling out of bounds, becuase the projection hasn't been retrained yet, 
        # - and set it to the boundary value
        if v[i] < low:
            v[i] = low
        elif v[i] > high:
            v[i] = high
        
        # Map the value v[i] to a cell index
        step = (high - low) / cells
        index[i] = int((v[i] - low) / step)
        
        if index[i] == cells: # In case it's right on the upper boundary
            index[i] -= 1

    return tuple(index)

# local test: 

# bounds = [(0, 1, 5), (0, 1, 5), (0, 1, 5)]  # 3-dimension grid with each dimension having range from 0 to 1 and 5 cells
# v = (0.1, 0.2, 0.9)  # The 3D vector

# index = vector_to_index(bounds, v)
# print(f"The index of vector {v} within the discrete grid is {index}")