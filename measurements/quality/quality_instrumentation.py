from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D
import numpy as np

# @inproceedings{alonso2020tensorflow,
#   title={Tensorflow Audio Models in {Essentia}},
#   author={Alonso-Jim{\'e}nez, Pablo and Bogdanov, Dmitry and Pons, Jordi and Serra, Xavier},
#   booktitle={International Conference on Acoustics, Speech and Signal Processing ({ICASSP})},
#   year={2020}
# }

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

def nsynth_instrument_topscore_and_index_and_class(audio_data, models_path):
  NSYNTH_EFFNET_MODEL_PATH=f"{models_path}/discogs-effnet-bs64-1.pb"
  embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=NSYNTH_EFFNET_MODEL_PATH, output="PartitionedCall:1")
  embeddings = embedding_model(audio_data)
  model = TensorflowPredict2D(graphFilename=f"{models_path}/nsynth_instrument-discogs-effnet-1.pb", output="model/Softmax")
  predictions = model(embeddings)
  predictions = np.mean(predictions, axis=0)
  # get the index of the highest probability
  index = np.argmax(predictions)
  # get the highest probability
  top_score = predictions[index]
  class_labels = ["mallet", "string", "reed", "guitar", "synth_lead", "vocal", "bass", "flute", "keyboard", "brass", "organ"]
  top_score_class = class_labels[index]
  return top_score, index, top_score_class