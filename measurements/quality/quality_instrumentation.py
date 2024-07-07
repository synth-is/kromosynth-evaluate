from essentia.standard import TensorflowPredictEffnetDiscogs, TensorflowPredict2D, TensorflowPredictVGGish
import numpy as np

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

def nsynth_instrument_predictions(audio_data, models_path):
  NSYNTH_EFFNET_MODEL_PATH=f"{models_path}/discogs-effnet-bs64-1.pb"
  embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=NSYNTH_EFFNET_MODEL_PATH, output="PartitionedCall:1")
  embeddings = embedding_model(audio_data)
  model = TensorflowPredict2D(graphFilename=f"{models_path}/nsynth_instrument-discogs-effnet-1.pb", output="model/Softmax")
  predictions = model(embeddings)
  return predictions

def yamnet_tagged_predictions(audio_data, models_path):
  model = TensorflowPredictVGGish(graphFilename=f"{models_path}/audioset-yamnet-1.pb", input="melspectrogram", output="activations")
  predictions = model(audio_data)
  predictions_dict = {}
  for i in range(len(yamnet_class_labels)):
    predictions_dict["YAM_" + yamnet_class_labels[i]] = predictions[0][i].astype(float)
  return predictions_dict

def mtg_jamendo_instrument_predictions(audio_data, models_path):
  embedding_model = TensorflowPredictEffnetDiscogs(graphFilename=f"{models_path}/discogs-effnet-bs64-1.pb", output="PartitionedCall:1")
  embeddings = embedding_model(audio_data)
  model = TensorflowPredict2D(graphFilename=f"{models_path}/mtg_jamendo_instrument-discogs-effnet-1.pb")
  predictions = model(embeddings)
  predictions_dict = {}
  for i in range(len(mtg_jamendo_instrument_class_labels)):
    predictions_dict["MTG_" + mtg_jamendo_instrument_class_labels[i]] = predictions[0][i].astype(float)
  return predictions_dict

# returns a dictionary of predictions
def nsynth_tagged_predictions(audio_data, models_path):
  global nsynth_class_labels
  predictions = nsynth_instrument_predictions(audio_data, models_path)
  predictions_dict = {}
  for i in range(len(nsynth_class_labels)):
    predictions_dict["NSY_" + nsynth_class_labels[i]] = predictions[0][i].astype(float)
  return predictions_dict



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