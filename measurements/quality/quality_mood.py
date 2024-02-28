# https://essentia.upf.edu/models.html 

from essentia.standard import TensorflowPredictMusiCNN, TensorflowPredict2D

# TODO preload the models? https://chromium.googlesource.com/external/github.com/tensorflow/tensorflow/+/r0.7/tensorflow/g3doc/how_tos/reading_data/index.md#Preloaded-data

def get_mood_predictions(mood_key, audio_data, models_path):
  MSD_MUSICNN_MODEL_PATH=f"{models_path}/msd-musicnn-1.pb"
  embedding_model = TensorflowPredictMusiCNN(graphFilename=MSD_MUSICNN_MODEL_PATH, output="model/dense/BiasAdd")
  embeddings = embedding_model(audio_data)
  model = TensorflowPredict2D(graphFilename=f"{models_path}/mood_{mood_key}-msd-musicnn-1.pb", output="model/Softmax")
  predictions = model(embeddings)
  return predictions

def mood_aggressive(audio_data, models_path):
  predictions = get_mood_predictions('aggressive', audio_data, models_path)
  # get the first column of the predictions, averaged to a single value
  mood_aggressive_averaged = sum(predictions[:,0]) / len(predictions[:,0])
  return mood_aggressive_averaged

def mood_happy(audio_data, models_path):
  predictions = get_mood_predictions('happy', audio_data, models_path)
  # get the first column of the predictions, averaged to a single value
  mood_happy_averaged = sum(predictions[:,0]) / len(predictions[:,0])
  return mood_happy_averaged

def mood_non_happy(audio_data, models_path):
  predictions = get_mood_predictions('happy', audio_data, models_path)
  # get the second column of the predictions, averaged to a single value
  mood_non_happy_averaged = sum(predictions[:,1]) / len(predictions[:,1])
  return mood_non_happy_averaged

def mood_party(audio_data, models_path):
  predictions = get_mood_predictions('party', audio_data, models_path)
  # get the first column of the predictions, averaged to a single value
  mood_party_averaged = sum(predictions[:,0]) / len(predictions[:,0])
  return mood_party_averaged

def mood_relaxed(audio_data, models_path):
  predictions = get_mood_predictions('relaxed', audio_data, models_path)
  # get the first column of the predictions, averaged to a single value
  mood_relaxed_averaged = sum(predictions[:,0]) / len(predictions[:,0])
  return mood_relaxed_averaged

def mood_sad(audio_data, models_path):
  predictions = get_mood_predictions('sad', audio_data, models_path)
  # get the first column of the predictions, averaged to a single value
  mood_sad_averaged = sum(predictions[:,0]) / len(predictions[:,0])
  return mood_sad_averaged

def mood_acoustic(audio_data, models_path):
  predictions = get_mood_predictions('acoustic', audio_data, models_path)
  # get the first column of the predictions, averaged to a single value
  mood_acoustic_averaged = sum(predictions[:,0]) / len(predictions[:,0])
  return mood_acoustic_averaged

def mood_electronic(audio_data, models_path):
  predictions = get_mood_predictions('electronic', audio_data, models_path)
  # get the first column of the predictions, averaged to a single value
  mood_electronic_averaged = sum(predictions[:,0]) / len(predictions[:,0])
  return mood_electronic_averaged

def get_top_mood(audio_data, models_path):
  mood_aggressive_averaged = mood_aggressive(audio_data, models_path)
  mood_happy_averaged = mood_happy(audio_data, models_path)
  mood_non_happy_averaged = mood_non_happy(audio_data, models_path)
  mood_party_averaged = mood_party(audio_data, models_path)
  mood_relaxed_averaged = mood_relaxed(audio_data, models_path)
  mood_sad_averaged = mood_sad(audio_data, models_path)
  mood_acoustic_averaged = mood_acoustic(audio_data, models_path)
  mood_electronic_averaged = mood_electronic(audio_data, models_path)
  mood_values = {
    "aggressive": mood_aggressive_averaged,
    "happy": mood_happy_averaged,
    "non_happy": mood_non_happy_averaged,
    "party": mood_party_averaged,
    "relaxed": mood_relaxed_averaged,
    "sad": mood_sad_averaged,
    "acoustic": mood_acoustic_averaged,
    "electronic": mood_electronic_averaged
  }
  # print('mood_values', mood_values)
  top_mood = max(mood_values, key=mood_values.get)
  return top_mood, mood_values[top_mood]