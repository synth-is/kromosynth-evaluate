from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score
# from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

def calculate_diversity_metrics(feature_vectors):
    # Behavioral Diversity
    behavioral_distances = pdist(feature_vectors, metric='cosine')
    behavioral_distances = behavioral_distances[~np.isnan(behavioral_distances)]
    # Normalize distances
    if behavioral_distances.size > 0:
        min_dist = np.min(behavioral_distances)
        max_dist = np.max(behavioral_distances)
        if max_dist - min_dist != 0:
            behavioral_distances = (behavioral_distances - min_dist) / (max_dist - min_dist)
        else:
            behavioral_distances = np.zeros_like(behavioral_distances)
    else:
        behavioral_distances = np.zeros_like(behavioral_distances)
    if behavioral_distances.size > 0:
        behavioral_diversity = {
            'mean': np.mean(behavioral_distances),
            'median': np.median(behavioral_distances),
            'std': np.std(behavioral_distances)
        }
    else:
        behavioral_diversity = {
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0
        }


    # Novelty Metric
    k = min(15, len(feature_vectors)-1)  # number of nearest neighbors or equal to feature_vectors length, if they are less than 15
    # The following approach does not scale well...
    # novelty_scores = []
    # for i, fv in enumerate(feature_vectors):
    #     distances = [np.linalg.norm(np.array(fv) - np.array(other_fv)) for j, other_fv in enumerate(feature_vectors) if i != j]
    #     novelty_scores.append(np.mean(sorted(distances)[:k]))
    # normalize novelty scoress
    # novelty_scores = np.array(novelty_scores)
    # novelty_scores = (novelty_scores - np.min(novelty_scores)) / (np.max(novelty_scores) - np.min(novelty_scores))
    
    # novelty_scores = calculate_novelty_metric(feature_vectors, k=k)

    novelty_scores = calculate_novelty_metric(feature_vectors, k=k, metric='cosine')

    # Handle NaN values in novelty_scores
    if np.isnan(novelty_scores).all():
        novelty_scores = np.zeros_like(novelty_scores)

    novelty_score_stats = {
        'mean': np.nanmean(novelty_scores),
        'median': np.nanmedian(novelty_scores),
        'std': np.nanstd(novelty_scores)
    }


    return {
        'behavioral_diversity': behavioral_diversity,
        'novelty_scores': novelty_scores.tolist(),
        'novelty_score_stats': novelty_score_stats
    }

def perform_cluster_analysis(feature_vectors):
    # Determine the optimal number of clusters using silhouette score
    # - For Cluster-Elites we use an initial number of centroids kinit = 50 and  increase it by adding kincr = 5 more centroids at every generation,  resulting at 5k centroids at the final generation.
    max_clusters = 10  # You can adjust this value
    best_n_clusters = 2
    best_score = -1

    for n_clusters in range(2, max_clusters + 1):
      kmeans = KMeans(n_clusters=n_clusters)
      labels = kmeans.fit_predict(feature_vectors)
      score = silhouette_score(feature_vectors, labels)
      if score > best_score:
        best_score = score
        best_n_clusters = n_clusters

    # Fit KMeans with the best number of clusters
    kmeans = KMeans(n_clusters=best_n_clusters)
    labels = kmeans.fit_predict(feature_vectors)
    cluster_sizes = [np.sum(labels == i) for i in range(kmeans.n_clusters)]
    
    return {
        'n_clusters': int(kmeans.n_clusters), # conversion to avoid Error: Object of type int64 is not JSON serializable
        'cluster_sizes': [int(size) for size in cluster_sizes]
    }

def calculate_performance_spread(feature_vectors, fitness_values, classification_dimensions):
    if classification_dimensions is not None:
        grid_shape = classification_dimensions
    else:
        grid_shape = (100, 100)
    grid = np.zeros(grid_shape)
    
    # Normalize feature vectors to the range [0, grid_shape - 1]
    for fv, fitness in zip(feature_vectors, fitness_values):
        if fitness is None:
            fitness = 0.0
            
        # Ensure indices are integers and within bounds
        indices = []
        for i, dim in enumerate(fv[:len(grid_shape)]):
            idx = int(np.clip(dim, 0, grid_shape[i] - 1))
            indices.append(idx)
            
        # Handle different dimensionalities
        if len(grid_shape) == 2:
            current_value = grid[indices[0], indices[1]]
            grid[indices[0], indices[1]] = max(float(current_value), float(fitness))
        elif len(grid_shape) == 3:
            current_value = grid[indices[0], indices[1], indices[2]]
            grid[indices[0], indices[1], indices[2]] = max(float(current_value), float(fitness))

    return {
        'mean': float(np.mean(grid)),
        'std': float(np.std(grid)),
        'min': float(np.min(grid)),
        'max': float(np.max(grid))
    }

# def calculate_novelty_metric(feature_vectors, k=15):
#     # Convert feature_vectors to a numpy array if it's not already
#     X = np.array(feature_vectors)
    
#     # Build the KD-tree
#     tree = KDTree(X)
    
#     # Query the tree for k+1 nearest neighbors (including the point itself)
#     distances, _ = tree.query(X, k=k+1)
    
#     # Exclude the first column (distance to self) and calculate mean
#     novelty_scores = np.mean(distances[:, 1:], axis=1)
    
#     # normalize novelty scoress
#     novelty_scores = np.array(novelty_scores)
#     min_score = np.min(novelty_scores)
#     max_score = np.max(novelty_scores)
#     if max_score - min_score != 0:
#         novelty_scores = (novelty_scores - min_score) / (max_score - min_score)
#     else:
#         novelty_scores = np.zeros_like(novelty_scores)

#     return novelty_scores

# TODO keeping the above commented out, as we're trying out this alternative approach, offering cosine similarity as a metric:

def calculate_novelty_metric(feature_vectors, k=15, metric='euclidean'):
    # Convert feature_vectors to a numpy array if it's not already
    X = np.array(feature_vectors)
    
    if metric == 'euclidean':
        # Use NearestNeighbors with 'auto' algorithm for Euclidean distance
        nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='auto')
    elif metric == 'cosine':
        # Normalize the vectors for cosine similarity
        X = normalize(X, axis=1, norm='l2')
        # Use NearestNeighbors with 'auto' algorithm for cosine similarity
        nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='auto')
    else:
        raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")
    
    # Fit and query
    nn.fit(X)
    distances, _ = nn.kneighbors(X)
    
    # Exclude the first column (distance to self) and calculate mean
    novelty_scores = np.mean(distances[:, 1:], axis=1)
    
    # Normalize novelty scores
    min_score = np.min(novelty_scores)
    max_score = np.max(novelty_scores)
    if max_score - min_score != 0:
        novelty_scores = (novelty_scores - min_score) / (max_score - min_score)
    else:
        novelty_scores = np.zeros_like(novelty_scores)

    return novelty_scores