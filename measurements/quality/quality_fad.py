from frechet_audio_distance import FrechetAudioDistance
import numpy as np
import time
from scipy.spatial import distance

# singleton frechet instance
frechet = None

maximum_distance = 0.0

maximum_euclidean_distance = 0.0

def get_fad_score(embeddings, background_embds_path, ckpt_dir):
    global frechet
    global maximum_distance
    if frechet is None:
        print('Creating new FrechetAudioDistance instance')
        frechet = FrechetAudioDistance(
            ckpt_dir=ckpt_dir,
            model_name="vggish", # not using the model, as the embeddings are already extracted, but the current implemention requires it to be set: TODO: change that?
            sample_rate=16000, # not using the sample_rate, as the embeddings are already extracted, but the current implemention requires it to be set: TODO: change that?
            verbose=True,
            audio_load_worker=8,
        )
    start_time = time.time()
    score = frechet.score_eval_embeddings(
        embeddings, 
        background_embds_path, 
        True # cache_background_embeds_in_memory
    )
    print(f"Time to calculate FAD: {time.time() - start_time}")

    if score > maximum_distance:
        maximum_distance = score
        print(f"--------- New maximum FAD score distance: {maximum_distance}")
    
    print('FAD score:', score)

    # invert the score, so that higher scores are better
    # score = 1 - score   
    # score = 1 / score

    score = rescale_distance(score)

    print('rescaled FAD score:', score)

    return score

# TODO: test further and maybe move into another module
def get_eucid_distance(embeddings, background_embds_path):
    global maximum_euclidean_distance
    # take the mean of the embeddings
    mean_embedding = np.mean(embeddings, axis=0)
    # print the mean embedding shape
    print('--- get_eucid_distance mean_embedding.shape', mean_embedding.shape)
    # load the background embeddings
    background_embeddings = np.load(background_embds_path)
    print('background_embeddings.shape', background_embeddings.shape)
    
    # for each background embedding, take the mean
    # TODO currently all frames from all sounds are a single sequence of vectors, as stored in the .npy files prepared by frechet-audio-distance, 
    # which is not suitable for the Euclidean calculation (right?), where we would rather like to take the mean of the frames for one sound at a time
    background_mean_embeddings = np.mean(background_embeddings, axis=0)
    
    print('--- get_eucid_distance background_mean_embeddings.shape', background_mean_embeddings.shape)
    # calculate the distance between the mean embedding and each background mean embeddings
    distances = distance.cdist([mean_embedding], [background_mean_embeddings], 'euclidean')
    print('--- get_eucid_distance distances.shape', distances.shape)
    print('--- get_eucid_distance distances', distances)
    # take the minimum distance
    min_distance = np.min(distances)
    if min_distance > maximum_euclidean_distance:
        maximum_euclidean_distance = min_distance
        print(f"--------- New maximum euclidean distance: {maximum_euclidean_distance}")
    print('get_eucid_distance min_distance', min_distance)
    # invert the distance, so that higher scores are better
    score = 1 / min_distance
    print('--- get_eucid_distance score', score)
    return score


# def rescale_fad_score(fad_score, upper_bound=50):
#     # Define the smallest_value based on expected FAD scores greater than zero.
#     # This value should be the smallest FAD score that you consider as "almost perfect".
#     smallest_value = 1e-11
    
#     # Handle edge cases where the FAD score is 0 or smaller than the smallest_value
#     if fad_score <= smallest_value:
#         return 1.0

#     # Handle edge cases where the FAD score is larger than or equal to the upper bound
#     if fad_score >= upper_bound:
#         return 0.0

#     # Linearly rescale the FAD score from (smallest_value, upper_bound) to (1, 0)
#     normalized_score = 1.0 - (fad_score - smallest_value) / (upper_bound - smallest_value)

#     return normalized_score

def rescale_distance(distance, max_distance=100):
    scaled_distance = distance / max_distance

    # Apply exponential decay to the scaled distance
    decayed_score = np.exp(-5 * scaled_distance)

    return decayed_score
