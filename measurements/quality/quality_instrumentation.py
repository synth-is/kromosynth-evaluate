from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D, TensorflowPredictVGGish, TensorflowPredictMusiCNN
import numpy as np
import time

# @inproceedings{alonso2020tensorflow,
#   title={Tensorflow Audio Models in {Essentia}},
#   author={Alonso-Jim{\'e}nez, Pablo and Bogdanov, Dmitry and Pons, Jordi and Serra, Xavier},
#   booktitle={International Conference on Acoustics, Speech and Signal Processing ({ICASSP})},
#   year={2020}
# }

# https://essentia.upf.edu/models.html#nsynth-instrument

nsynth_class_labels = ["mallet", "string", "reed", "guitar", "synth_lead", "vocal", "bass", "flute", "keyboard", "brass", "organ"]
yamnet_class_labels = ["Speech","Child speech, kid speaking","Conversation","Narration, monologue","Babbling","Speech synthesizer","Shout","Bellow","Whoop","Yell","Children shouting","Screaming","Whispering","Laughter","Baby laughter","Giggle","Snicker","Belly laugh","Chuckle, chortle","Crying, sobbing","Baby cry, infant cry","Whimper","Wail, moan","Sigh","Singing","Choir","Yodeling","Chant","Mantra","Child singing","Synthetic singing","Rapping","Humming","Groan","Grunt","Whistling","Breathing","Wheeze","Snoring","Gasp","Pant","Snort","Cough","Throat clearing","Sneeze","Sniff","Run","Shuffle","Walk, footsteps","Chewing, mastication","Biting","Gargling","Stomach rumble","Burping, eructation","Hiccup","Fart","Hands","Finger snapping","Clapping","Heart sounds, heartbeat","Heart murmur","Cheering","Applause","Chatter","Crowd","Hubbub, speech noise, speech babble","Children playing","Animal","Domestic animals, pets","Dog","Bark","Yip","Howl","Bow-wow","Growling","Whimper (dog)","Cat","Purr","Meow","Hiss","Caterwaul","Livestock, farm animals, working animals","Horse","Clip-clop","Neigh, whinny","Cattle, bovinae","Moo","Cowbell","Pig","Oink","Goat","Bleat","Sheep","Fowl","Chicken, rooster","Cluck","Crowing, cock-a-doodle-doo","Turkey","Gobble","Duck","Quack","Goose","Honk","Wild animals","Roaring cats (lions, tigers)","Roar","Bird","Bird vocalization, bird call, bird song","Chirp, tweet","Squawk","Pigeon, dove","Coo","Crow","Caw","Owl","Hoot","Bird flight, flapping wings","Canidae, dogs, wolves","Rodents, rats, mice","Mouse","Patter","Insect","Cricket","Mosquito","Fly, housefly","Buzz","Bee, wasp, etc.","Frog","Croak","Snake","Rattle","Whale vocalization","Music","Musical instrument","Plucked string instrument","Guitar","Electric guitar","Bass guitar","Acoustic guitar","Steel guitar, slide guitar","Tapping (guitar technique)","Strum","Banjo","Sitar","Mandolin","Zither","Ukulele","Keyboard (musical)","Piano","Electric piano","Organ","Electronic organ","Hammond organ","Synthesizer","Sampler","Harpsichord","Percussion","Drum kit","Drum machine","Drum","Snare drum","Rimshot","Drum roll","Bass drum","Timpani","Tabla","Cymbal","Hi-hat","Wood block","Tambourine","Rattle (instrument)","Maraca","Gong","Tubular bells","Mallet percussion","Marimba, xylophone","Glockenspiel","Vibraphone","Steelpan","Orchestra","Brass instrument","French horn","Trumpet","Trombone","Bowed string instrument","String section","Violin, fiddle","Pizzicato","Cello","Double bass","Wind instrument, woodwind instrument","Flute","Saxophone","Clarinet","Harp","Bell","Church bell","Jingle bell","Bicycle bell","Tuning fork","Chime","Wind chime","Change ringing (campanology)","Harmonica","Accordion","Bagpipes","Didgeridoo","Shofar","Theremin","Singing bowl","Scratching (performance technique)","Pop music","Hip hop music","Beatboxing","Rock music","Heavy metal","Punk rock","Grunge","Progressive rock","Rock and roll","Psychedelic rock","Rhythm and blues","Soul music","Reggae","Country","Swing music","Bluegrass","Funk","Folk music","Middle Eastern music","Jazz","Disco","Classical music","Opera","Electronic music","House music","Techno","Dubstep","Drum and bass","Electronica","Electronic dance music","Ambient music","Trance music","Music of Latin America","Salsa music","Flamenco","Blues","Music for children","New-age music","Vocal music","A capella","Music of Africa","Afrobeat","Christian music","Gospel music","Music of Asia","Carnatic music","Music of Bollywood","Ska","Traditional music","Independent music","Song","Background music","Theme music","Jingle (music)","Soundtrack music","Lullaby","Video game music","Christmas music","Dance music","Wedding music","Happy music","Sad music","Tender music","Exciting music","Angry music","Scary music","Wind","Rustling leaves","Wind noise (microphone)","Thunderstorm","Thunder","Water","Rain","Raindrop","Rain on surface","Stream","Waterfall","Ocean","Waves, surf","Steam","Gurgling","Fire","Crackle","Vehicle","Boat, Water vehicle","Sailboat, sailing ship","Rowboat, canoe, kayak","Motorboat, speedboat","Ship","Motor vehicle (road)","Car","Vehicle horn, car horn, honking","Toot","Car alarm","Power windows, electric windows","Skidding","Tire squeal","Car passing by","Race car, auto racing","Truck","Air brake","Air horn, truck horn","Reversing beeps","Ice cream truck, ice cream van","Bus","Emergency vehicle","Police car (siren)","Ambulance (siren)","Fire engine, fire truck (siren)","Motorcycle","Traffic noise, roadway noise","Rail transport","Train","Train whistle","Train horn","Railroad car, train wagon","Train wheels squealing","Subway, metro, underground","Aircraft","Aircraft engine","Jet engine","Propeller, airscrew","Helicopter","Fixed-wing aircraft, airplane","Bicycle","Skateboard","Engine","Light engine (high frequency)","Dental drill, dentist's drill","Lawn mower","Chainsaw","Medium engine (mid frequency)","Heavy engine (low frequency)","Engine knocking","Engine starting","Idling","Accelerating, revving, vroom","Door","Doorbell","Ding-dong","Sliding door","Slam","Knock","Tap","Squeak","Cupboard open or close","Drawer open or close","Dishes, pots, and pans","Cutlery, silverware","Chopping (food)","Frying (food)","Microwave oven","Blender","Water tap, faucet","Sink (filling or washing)","Bathtub (filling or washing)","Hair dryer","Toilet flush","Toothbrush","Electric toothbrush","Vacuum cleaner","Zipper (clothing)","Keys jangling","Coin (dropping)","Scissors","Electric shaver, electric razor","Shuffling cards","Typing","Typewriter","Computer keyboard","Writing","Alarm","Telephone","Telephone bell ringing","Ringtone","Telephone dialing, DTMF","Dial tone","Busy signal","Alarm clock","Siren","Civil defense siren","Buzzer","Smoke detector, smoke alarm","Fire alarm","Foghorn","Whistle","Steam whistle","Mechanisms","Ratchet, pawl","Clock","Tick","Tick-tock","Gears","Pulleys","Sewing machine","Mechanical fan","Air conditioning","Cash register","Printer","Camera","Single-lens reflex camera","Tools","Hammer","Jackhammer","Sawing","Filing (rasp)","Sanding","Power tool","Drill","Explosion","Gunshot, gunfire","Machine gun","Fusillade","Artillery fire","Cap gun","Fireworks","Firecracker","Burst, pop","Eruption","Boom","Wood","Chop","Splinter","Crack","Glass","Chink, clink","Shatter","Liquid","Splash, splatter","Slosh","Squish","Drip","Pour","Trickle, dribble","Gush","Fill (with liquid)","Spray","Pump (liquid)","Stir","Boiling","Sonar","Arrow","Whoosh, swoosh, swish","Thump, thud","Thunk","Electronic tuner","Effects unit","Chorus effect","Basketball bounce","Bang","Slap, smack","Whack, thwack","Smash, crash","Breaking","Bouncing","Whip","Flap","Scratch","Scrape","Rub","Roll","Crushing","Crumpling, crinkling","Tearing","Beep, bleep","Ping","Ding","Clang","Squeal","Creak","Rustle","Whir","Clatter","Sizzle","Clicking","Clickety-clack","Rumble","Plop","Jingle, tinkle","Hum","Zing","Boing","Crunch","Silence","Sine wave","Harmonic","Chirp tone","Sound effect","Pulse","Inside, small room","Inside, large room or hall","Inside, public space","Outside, urban or manmade","Outside, rural or natural","Reverberation","Echo","Noise","Environmental noise","Static","Mains hum","Distortion","Sidetone","Cacophony","White noise","Pink noise","Throbbing","Vibration","Television","Radio","Field recording"];
mtg_jamendo_instrument_class_labels = ["accordion", "acousticbassguitar", "acousticguitar", "bass", "beat", "bell", "bongo", "brass", "cello", "clarinet", "classicalguitar", "computer", "doublebass", "drummachine", "drums", "electricguitar", "electricpiano", "flute", "guitar", "harmonica", "harp", "horn", "keyboard", "oboe", "orchestra", "organ", "pad", "percussion", "piano", "pipeorgan", "rhodes", "sampler", "saxophone", "strings", "synthesizer", "trombone", "trumpet", "viola", "violin", "voice"]
music_loop_instrument_role_class_labels = ["bass", "chords", "fx", "melody", "percussion"]
mood_acoustic_class_labels = ["acoustic", "non_acoustic"]
mood_electronic_class_labels = ["electronic", "non_electronic"]
voice_instrumental_class_labels = ["instrumental", "voice"]
voice_gender_class_labels = ["female", "male"]
timbre_class_labels = ["bright", "dark"]
nsynth_acoustic_electronic_class_labels = ["acoustic", "electronic"]
nsynth_bright_dark_class_labels = ["bright", "dark"]
nsynth_reverb_class_labels = ["dry", "wet"]

models = {}

##### predictions

### instrumentation

# Nsynth instrument
def nsynth_instrument_predictions(audio_data, models_path):
  NSYNTH_EFFNET_MODEL_PATH=f"{models_path}/discogs-effnet-bs64-1.pb"
  if 'discogs-effnet-bs64-1_model' not in models:
    models['discogs-effnet-bs64-1_model'] = TensorflowPredictEffnetDiscogs(graphFilename=NSYNTH_EFFNET_MODEL_PATH, output="PartitionedCall:1")
  embedding_model = models['discogs-effnet-bs64-1_model']
  embeddings = embedding_model(audio_data)
  if 'nsynth_model' not in models:
    models['nsynth_model'] = TensorflowPredict2D(graphFilename=f"{models_path}/nsynth_instrument-discogs-effnet-1.pb", output="model/Softmax")
  model = models['nsynth_model']
  predictions = model(embeddings)
  return predictions

# returns a dictionary of predictions
def nsynth_tagged_predictions(audio_data, models_path):
  start_time = time.time()
  global nsynth_class_labels
  predictions = nsynth_instrument_predictions(audio_data, models_path)
  predictions_dict = {}
  for i in range(len(nsynth_class_labels)):
    predictions_dict["NSY_" + nsynth_class_labels[i]] = predictions[0][i].astype(float)
  end_time = time.time()
  print('nsynth_tagged_predictions: Time taken to evaluate fitness:', end_time - start_time)
  return predictions_dict

# MTG-Jamendo instrument
def mtg_jamendo_instrument_predictions(audio_data, models_path):
  start_time = time.time()
  if 'discogs-effnet-bs64-1_model' not in models:
    models['discogs-effnet-bs64-1_model'] = TensorflowPredictEffnetDiscogs(graphFilename=f"{models_path}/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
  embedding_model = models['discogs-effnet-bs64-1_model']
  embeddings = embedding_model(audio_data)
  if 'mtg_jamendo_instrument_model' not in models:
    models['mtg_jamendo_instrument_model'] = TensorflowPredict2D(graphFilename=f"{models_path}/mtg_jamendo_instrument-discogs-effnet-1.pb")
  model = models['mtg_jamendo_instrument_model']
  predictions = model(embeddings)
  predictions_dict = {}
  for i in range(len(mtg_jamendo_instrument_class_labels)):
    predictions_dict["MTG_" + mtg_jamendo_instrument_class_labels[i]] = predictions[0][i].astype(float)
  end_time = time.time()
  print('mtg_jamendo_instrument_predictions: Time taken to evaluate fitness:', end_time - start_time)
  return predictions_dict

# Music loop instrument role
# Classification of music loops by their instrument role using the Freesound Loop Dataset (5 classes):
def music_loop_instrument_role_predictions(audio_data, models_path):
  start_time = time.time()
  if 'msd-musicnn-1_model' not in models:
    models['msd-musicnn-1_model'] = TensorflowPredictMusiCNN(graphFilename=f"{models_path}/msd-musicnn-1.pb", output="model/dense/BiasAdd")
  embedding_model = models['msd-musicnn-1_model']
  embeddings = embedding_model(audio_data)
  if 'music_loop_instrument_model' not in models:
    models['music_loop_instrument_model'] = TensorflowPredict2D(graphFilename=f"{models_path}/fs_loop_ds-msd-musicnn-1.pb", input="serving_default_model_Placeholder", output="PartitionedCall")
  model = models['music_loop_instrument_model']
  predictions = model(embeddings)
  predictions_dict = {}
  for i in range(len(music_loop_instrument_role_class_labels)):
    predictions_dict["MLIR_" + music_loop_instrument_role_class_labels[i]] = predictions[0][i].astype(float)
  end_time = time.time()
  print('music_loop_instrument_role_predictions: Time taken to evaluate fitness:', end_time - start_time)
  return predictions_dict

#Mood Acoustic
#Music classification by type of sound (2 classes):
def mood_acoustic_predictions(audio_data, models_path):
  start_time = time.time()
  if 'audioset-vggish-3_model' not in models:
    models['audioset-vggish-3_model'] = TensorflowPredictVGGish(graphFilename=f"{models_path}/audioset-vggish-3.pb", output="model/vggish/embeddings")
  embedding_model = models['audioset-vggish-3_model']
  embeddings = embedding_model(audio_data)
  if 'mood_acoustic_model' not in models:
    models['mood_acoustic_model'] = TensorflowPredict2D(graphFilename=f"{models_path}/mood_acoustic-audioset-vggish-1.pb", output="model/Softmax")
  model = models['mood_acoustic_model']
  predictions = model(embeddings)
  predictions_dict = {}
  for i in range(len(mood_acoustic_class_labels)):
    predictions_dict["MA_" + mood_acoustic_class_labels[i]] = predictions[0][i].astype(float)
  end_time = time.time()
  print('mood_acoustic_predictions: Time taken to evaluate fitness:', end_time - start_time)
  return predictions_dict

# Mood Electronic
# Music classification by type of sound (2 classes):
def mood_electronic_predictions(audio_data, models_path):
  start_time = time.time()
  if 'audioset-vggish-3_model' not in models:
    models['audioset-vggish-3_model'] = TensorflowPredictVGGish(graphFilename=f"{models_path}/audioset-vggish-3.pb", output="model/vggish/embeddings")
  embedding_model = models['audioset-vggish-3_model']
  embeddings = embedding_model(audio_data)
  if 'mood_electronic_model' not in models:
    models['mood_electronic_model'] = TensorflowPredict2D(graphFilename=f"{models_path}/mood_electronic-audioset-vggish-1.pb", output="model/Softmax")
  model = models['mood_electronic_model']
  predictions = model(embeddings)
  predictions_dict = {}
  for i in range(len(mood_electronic_class_labels)):
    predictions_dict["ME_" + mood_electronic_class_labels[i]] = predictions[0][i].astype(float)
  end_time = time.time()
  print('mood_electronic_predictions: Time taken to evaluate fitness:', end_time - start_time)
  return predictions_dict

# Voice/instrumental
#Classification of music by presence or absence of voice (2 classes):
def voice_instrumental_predictions(audio_data, models_path):
  start_time = time.time()
  if 'audioset-vggish-3_model' not in models:
    models['audioset-vggish-3_model'] = TensorflowPredictVGGish(graphFilename=f"{models_path}/audioset-vggish-3.pb", output="model/vggish/embeddings")
  embedding_model = models['audioset-vggish-3_model']
  embeddings = embedding_model(audio_data)
  if 'voice_instrumental_model' not in models:
    models['voice_instrumental_model'] = TensorflowPredict2D(graphFilename=f"{models_path}/voice_instrumental-audioset-vggish-1.pb", output="model/Softmax")
  model = models['voice_instrumental_model']
  predictions = model(embeddings)
  predictions_dict = {}
  for i in range(len(voice_instrumental_class_labels)):
    predictions_dict["VI_" + voice_instrumental_class_labels[i]] = predictions[0][i].astype(float)
  end_time = time.time()
  print('voice_instrumental_predictions: Time taken to evaluate fitness:', end_time - start_time)
  return predictions_dict

# Voice gender
#Classification of music by singing voice gender (2 classes):
def voice_gender_predictions(audio_data, models_path):
  start_time = time.time()
  if 'audioset-vggish-3_model' not in models:
    models['audioset-vggish-3_model'] = TensorflowPredictVGGish(graphFilename=f"{models_path}/audioset-vggish-3.pb", output="model/vggish/embeddings")
  embedding_model = models['audioset-vggish-3_model']
  embeddings = embedding_model(audio_data)
  if 'voice_gender_model' not in models:
    models['voice_gender_model'] = TensorflowPredict2D(graphFilename=f"{models_path}/gender-audioset-vggish-1.pb", output="model/Softmax")
  model = models['voice_gender_model']
  predictions = model(embeddings)
  predictions_dict = {}
  for i in range(len(voice_gender_class_labels)):
    predictions_dict["VG_" + voice_gender_class_labels[i]] = predictions[0][i].astype(float)
  end_time = time.time()
  print('voice_gender_predictions: Time taken to evaluate fitness:', end_time - start_time)
  return predictions_dict

# Timbre
# Classification of music by timbre color (2 classes):
def timbre_predictions(audio_data, models_path):
  start_time = time.time()
  if 'discogs-effnet-bs64-1_model' not in models:
    models['discogs-effnet-bs64-1_model'] = TensorflowPredictEffnetDiscogs(graphFilename=f"{models_path}/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
  embedding_model = models['discogs-effnet-bs64-1_model']
  embeddings = embedding_model(audio_data)
  if 'timbre_model' not in models:
    models['timbre_model'] = TensorflowPredict2D(graphFilename=f"{models_path}/timbre-discogs-effnet-1.pb", output="model/Softmax")
  model = models['timbre_model']
  predictions = model(embeddings)
  predictions_dict = {}
  for i in range(len(timbre_class_labels)):
    predictions_dict["TIM_" + timbre_class_labels[i]] = predictions[0][i].astype(float)
  end_time = time.time()
  print('timbre_predictions: Time taken to evaluate fitness:', end_time - start_time)
  return predictions_dict

# Nsynth acoustic/electronic
# Classification of monophonic sources into acoustic or electronic origin using the Nsynth dataset (2 classes):
def nsynth_acoustic_electronic_predictions(audio_data, models_path):
  start_time = time.time()
  if 'discogs-effnet-bs64-1_model' not in models:
    models['discogs-effnet-bs64-1_model'] = TensorflowPredictEffnetDiscogs(graphFilename=f"{models_path}/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
  embedding_model = models['discogs-effnet-bs64-1_model']
  embeddings = embedding_model(audio_data)
  if 'nsynth_acoustic_electronic_model' not in models:
    models['nsynth_acoustic_electronic_model'] = TensorflowPredict2D(graphFilename=f"{models_path}/nsynth_acoustic_electronic-discogs-effnet-1.pb", output="model/Softmax")
  model = models['nsynth_acoustic_electronic_model']
  predictions = model(embeddings)
  predictions_dict = {}
  for i in range(len(nsynth_acoustic_electronic_class_labels)):
    predictions_dict["NAE_" + nsynth_acoustic_electronic_class_labels[i]] = predictions[0][i].astype(float)
  end_time = time.time()
  print('nsynth_acoustic_electronic_predictions: Time taken to evaluate fitness:', end_time - start_time)
  return predictions_dict

# Nsynth bright/dark
# Classification of monophonic sources by timbre color using the Nsynth dataset (2 classes):
def nsynth_bright_dark_predictions(audio_data, models_path):
  start_time = time.time()
  if 'discogs-effnet-bs64-1_model' not in models:
    models['discogs-effnet-bs64-1_model'] = TensorflowPredictEffnetDiscogs(graphFilename=f"{models_path}/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
  embedding_model = models['discogs-effnet-bs64-1_model']
  embeddings = embedding_model(audio_data)
  if 'nsynth_bright_dark_model' not in models:
    models['nsynth_bright_dark_model'] = TensorflowPredict2D(graphFilename=f"{models_path}/nsynth_bright_dark-discogs-effnet-1.pb", output="model/Softmax")
  model = models['nsynth_bright_dark_model']
  predictions = model(embeddings)
  predictions_dict = {}
  for i in range(len(nsynth_bright_dark_class_labels)):
    predictions_dict["NBD_" + nsynth_bright_dark_class_labels[i]] = predictions[0][i].astype(float)
  end_time = time.time()
  print('nsynth_bright_dark_predictions: Time taken to evaluate fitness:', end_time - start_time)
  return predictions_dict

# Nsynth reverb
# Detection of reverb in monophonic sources using the Nsynth dataset (2 classes):
def nsynth_reverb_predictions(audio_data, models_path):
  start_time = time.time()
  if 'discogs-effnet-bs64-1_model' not in models:
    models['discogs-effnet-bs64-1_model'] = TensorflowPredictEffnetDiscogs(graphFilename=f"{models_path}/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
  embedding_model = models['discogs-effnet-bs64-1_model']
  embeddings = embedding_model(audio_data)
  if 'nsynth_reverb_model' not in models:
    models['nsynth_reverb_model'] = TensorflowPredict2D(graphFilename=f"{models_path}/nsynth_reverb-discogs-effnet-1.pb", output="model/Softmax")
  model = models['nsynth_reverb_model']
  predictions = model(embeddings)
  predictions_dict = {}
  for i in range(len(nsynth_reverb_class_labels)):
    predictions_dict["NRV_" + nsynth_reverb_class_labels[i]] = predictions[0][i].astype(float)
  end_time = time.time()
  print('nsynth_reverb_predictions: Time taken to evaluate fitness:', end_time - start_time)
  return predictions_dict


### event recognition
def yamnet_tagged_predictions(audio_data, models_path):
  start_time = time.time()
  # is the model already loaded?
  if 'yamnet_model' not in models:
    models['yamnet_model'] = TensorflowPredictVGGish(graphFilename=f"{models_path}/audioset-yamnet-1.pb", input="melspectrogram", output="activations")
  model = models['yamnet_model']
  predictions = model(audio_data)
  predictions_dict = {}
  for i in range(len(yamnet_class_labels)):
    predictions_dict["YAM_" + yamnet_class_labels[i]] = predictions[0][i].astype(float)
  end_time = time.time()
  print('yamnet_tagged_predictions: Time taken to evaluate fitness:', end_time - start_time)
  return predictions_dict



### aggregations

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


