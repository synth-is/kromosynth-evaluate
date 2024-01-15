import numpy as np

# import numpy as np

# def find_index(n, x, v):
#     grid = np.zeros([x]*n)
#     shape = grid.shape
#     index = np.ravel_multi_index(v, shape)
#     return index

# def discretise_2D_array(array_2D, n):
    

#     # Get the shape of the original array
#     original_shape = np.array(array_2D.shape)
    
#     # Calculate the new shape
#     new_shape = original_shape//n * n
    
#     # Reshape the original array to the new shape
#     reshaped_array = array_2D[:new_shape[0], :new_shape[1]].reshape(-1, n, n)
    
#     # Calculate the mean of each cell and reshape it to the original shape
#     discretised_array = reshaped_array.mean(axis=(1,2)).reshape(original_shape[0]//n, original_shape[1]//n)
    
#     return discretised_array

# # Example 2D array
# example_data = np.array( [[1, 2, 3], [4, 5, 6], [7, 8, 9]] )

# # Call the function
# example_data = discretise_2D_array(example_data, 2)

# # Print the result
# print(example_data)

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
        if int(v[i]) < low or int(v[i]) > high: # int cast to avoid float comparison errors, e.g. v[i] can sometimes be 1.0000000000000002 instead of 1.0
            raise ValueError(f"Value v[{i}] is out of bounds.")
        
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