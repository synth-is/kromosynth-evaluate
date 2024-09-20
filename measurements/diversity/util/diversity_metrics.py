from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import numpy as np

def calculate_diversity_metrics(feature_vectors, genotypes):
    # Behavioral Diversity
    behavioral_distances = pdist(feature_vectors)
    behavioral_diversity = {
        'mean': np.mean(behavioral_distances),
        'median': np.median(behavioral_distances),
        'std': np.std(behavioral_distances)
    }

    # Genotypic Diversity
    genotypic_distances = pdist(genotypes)
    genotypic_diversity = {
        'mean': np.mean(genotypic_distances),
        'median': np.median(genotypic_distances),
        'std': np.std(genotypic_distances)
    }

    # Novelty Metric
    k = 15  # number of nearest neighbors
    novelty_scores = []
    for i, fv in enumerate(feature_vectors):
        distances = [np.linalg.norm(fv - other_fv) for j, other_fv in enumerate(feature_vectors) if i != j]
        novelty_scores.append(np.mean(sorted(distances)[:k]))

    return {
        'behavioral_diversity': behavioral_diversity,
        'genotypic_diversity': genotypic_diversity,
        'novelty_scores': novelty_scores
    }

def perform_cluster_analysis(feature_vectors):
    kmeans = KMeans(n_clusters=5)  # You might want to make this adaptive
    labels = kmeans.fit_predict(feature_vectors)
    cluster_sizes = [np.sum(labels == i) for i in range(kmeans.n_clusters)]
    
    return {
        'n_clusters': kmeans.n_clusters,
        'cluster_sizes': cluster_sizes
    }

def calculate_performance_spread(feature_vectors, fitness_values):
    # Assuming you're using a grid-based approach
    grid = np.zeros((10, 10))  # Adjust grid size as needed
    for fv, fitness in zip(feature_vectors, fitness_values):
        x, y = np.clip(fv * 10, 0, 9).astype(int)
        grid[x, y] = max(grid[x, y], fitness)

    return {
        'mean': np.mean(grid),
        'std': np.std(grid),
        'min': np.min(grid),
        'max': np.max(grid)
    }