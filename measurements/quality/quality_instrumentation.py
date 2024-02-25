from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D
import numpy as np

# https://essentia.upf.edu/models.html#nsynth-instrument
# returns the mean of the probabilities of the 11 instrument classes
def nsynth_instrument_mean(audio_data, models_path):
  NSYNTH_EFFNET_MODEL_PATH=f"{models_path}/discogs-effnet-bs64-1.pb"
  embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=NSYNTH_EFFNET_MODEL_PATH, output="PartitionedCall:1")
  embeddings = embedding_model(audio_data)
  model = TensorflowPredict2D(graphFilename=f"{models_path}/nsynth_instrument-discogs-effnet-1.pb", output="model/Softmax")
  predictions = model(embeddings)
  # predictions are an array of arrays, where each sub-array contains 11 elements, representing the probabilities of each musical instrument class
  # average the probabilities of the 11 classes
  predictions = np.mean(predictions, axis=0).mean()
  return predictions