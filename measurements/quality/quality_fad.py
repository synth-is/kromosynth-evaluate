from frechet_audio_distance import FrechetAudioDistance
import numpy as np
import time

# singleton frechet instance
frechet = None

maximum_distance = 0.0

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
