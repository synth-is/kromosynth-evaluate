from frechet_audio_distance import FrechetAudioDistance

def get_fad_score(embeddings, background_embds_path, ckpt_dir):
    frechet = FrechetAudioDistance(
        ckpt_dir=ckpt_dir,
        model_name="vggish", # not using the model, as the embeddings are already extracted, but the current implemention requires it to be set: TODO: change that?
        sample_rate=16000, # not using the sample_rate, as the embeddings are already extracted, but the current implemention requires it to be set: TODO: change that?
        verbose=True,
        audio_load_worker=8,
    )
    score = frechet.score_eval_embeddings(embeddings, background_embds_path)
    return score