# quality signals from "Automatic detection of audio problems for quality control":
# - https://www.aes.org/e-lib/browse.cfm?elib=20338
# - https://essentia.upf.edu/python_examples.html#audio-problems

from essentia.standard import (
   FrameGenerator, ClickDetector, DiscontinuityDetector, GapsDetector, HumDetector, SaturationDetector, SNR, TruePeakDetector, NoiseBurstDetector,
   Energy
)
from measurements.quality.util import normalize_and_clamp
import numpy as np
import zlib

def energy(audio_data):
  # energy is 80000.0 with a square wave
  energy = Energy()(audio_data)
  energy_normalized = normalize_and_clamp(energy, 80000.0)
  # print(f'energy_normalized: {energy_normalized}')
  return energy_normalized

def click_count_percentage(audio_data, sample_rate):
  # TODO: configure those via parameters?
  frame_size = 512
  hop_size = 256

  print(f'audio_data: {audio_data.shape}')

  clickDetector = ClickDetector(frameSize=frame_size, hopSize=hop_size)

  starts, ends = [], []
  click_count = 0
  for frame in FrameGenerator(audio_data, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
    frame_starts, frame_ends = clickDetector(frame)
    # print(f'frame_starts: {frame_starts}')
    # print(f'frame_ends: {frame_ends}')
    # if frame_starts and frame_ends are not empty
    if frame_starts.size > 0 and frame_ends.size > 0:
        # print(f'Frame starts: {frame_starts}')
        # print(f'Frame ends: {frame_ends}')
        click_count += 1 # TODO: should take into account multiple clicks per frame
    starts.extend(list(frame_starts))
    ends.extend(list(frame_ends))

  print(f'click_count: {click_count}')
  click_count_percentage = click_count / (len(audio_data) / hop_size)
  print(f'click_count_percentage: {click_count_percentage}')
  return click_count_percentage

def discontinuity_count_percentage(audio_data, sample_rate):
  frame_size = 512
  hop_size = 256

  discontinuityDetector = DiscontinuityDetector(frameSize=frame_size, hopSize=hop_size)

  locs = []
  for idx, frame in enumerate(
      FrameGenerator(audio_data, frameSize=frame_size, hopSize=hop_size, startFromZero=True)
  ):
      frame_locs, _ = discontinuityDetector(frame)
      locs.extend((frame_locs + hop_size * idx) / sample_rate)

  # number of detected discontinuities
  discontinuity_count = len(locs)
  print('Number of detected discontinuities: {}'.format(discontinuity_count))
  # discontinuity locations counts normalized by the number of frames
  discontinuity_count_percentage = discontinuity_count / len(audio_data) * sample_rate / hop_size
  print('Number of detected discontinuities per second: {}'.format(discontinuity_count_percentage))

  return discontinuity_count_percentage

def gaps_count_percentage(audio_data, sample_rate):
  frame_size = 512
  hop_size = 256

  gapDetector = GapsDetector(frameSize=frame_size, hopSize=hop_size)

  starts, ends = [], []
  for frame in FrameGenerator(
      audio_data, frameSize=frame_size, hopSize=hop_size, startFromZero=True
  ):
      frame_starts, frame_ends = gapDetector(frame)
      starts.extend(frame_starts)
      ends.extend(frame_ends)

  # proportion of frames that are detected as gaps, using starts and ends
  duration_gaps = sum(ends) - sum(starts)
  duration_audio = len(audio_data) / sample_rate
  proportion_gaps = duration_gaps / duration_audio
  print(f'Proportion of gaps: {proportion_gaps:.2f}')

  return proportion_gaps
      

def hum_precence_percentage(audio_data, sample_rate):
  time_window = 5
  r0, freqs, saliences, starts, ends = HumDetector(timeWindow=time_window)(audio_data)

  xmin = time_window
  xmax = time_window + r0.shape[1] * .2
  ymin = 0
  ymax = 1000  # The algorithm only searches up to 1kHz.


  # Plot the detected tones as horizontal red lines
  for i in range(len(freqs)):
      print(
          "Detected a hum at {:.2f}Hz with salience {:.2f} starting at {:.2f}s and ending at {:.2f}s"\
              .format(freqs[i], saliences[i], starts[i], ends[i]
          )
      )

  print("audio sample count: ", len(audio_data))
  print("sample rate: ", sample_rate)

  audio_duration = len(audio_data) / sample_rate
  print("Audio duration: {:.2f}s".format(audio_duration))
  # proportion of time the hum is present
  hum_presence = np.sum(ends - starts) / audio_duration
  print("Hum presence: {:.2f}%".format(hum_presence * 100))
  # hum_precense scaled by the salience
  hum_presence_weighted = np.sum((ends - starts) * saliences) / audio_duration
  print("Hum presence weighted by salience: {:.2f}%".format(hum_presence_weighted * 100))

  return hum_presence_weighted

def saturation_percentage(audio_data, sample_rate):
  frame_size = 512
  hop_size = 256

  saturationDetector = SaturationDetector(frameSize=frame_size,
                                          hopSize=hop_size)

  starts, ends = [], []
  for frame in FrameGenerator(audio_data,
                              frameSize=frame_size,
                              hopSize=hop_size,
                              startFromZero=True):
      frame_starts, frame_ends = saturationDetector(frame)
      starts.extend(frame_starts)
      ends.extend(frame_ends)
  # print(starts)
  # print(ends)

  # count the number of frames that are saturated
  num_saturated_frames = len(starts)
  print("num_saturated_frames:",num_saturated_frames)

  # duration in seconds that is saturated, by counting deltas between starts and ends
  duration_saturated = sum(ends) - sum(starts)
  print("duration_saturated:", duration_saturated)
  # total duration of the audio in seconds
  duration = len(audio_data) / sample_rate
  print("duration:", duration)
  # calculate the percentage of saturated audio
  percent_saturated = duration_saturated / duration
  print("percent_saturated:", percent_saturated)

  return percent_saturated

def signal_to_noise_percentage_of_excellence(audio_data, sample_rate):
  threshold = -40

  frame_size = 512

  broadbad_correction = True

  snr = SNR(
    frameSize=frame_size,
    noiseThreshold=threshold,
    useBroadbadNoiseCorrection=broadbad_correction,
  )

  snr_spectral_list = []
  for frame in FrameGenerator(
    audio_data,
    frameSize=frame_size,
    hopSize=frame_size // 2,
  ):
    snr_instant, snr_ema, snr_spetral = snr(frame)

  print("SNR EMA:", snr_ema)

  excellent_snr = 40
  # if snr_ema is infinite or nan, set it to 0
  if np.isinf(snr_ema) or np.isnan(snr_ema):
    snr_ema = excellent_snr
  snr_percentage_of_excellence = snr_ema / excellent_snr
  print("SNR percentage of excellence:", snr_percentage_of_excellence)

  return 1 - snr_percentage_of_excellence

def true_peak_clipping_percentage(audio_data, sample_rate):
  
  # the signal is oversampled by default with a factor of 4: https://essentia.upf.edu/reference/std_TruePeakDetector.html
  oversampling_factor = 4
  peak_locations, output = TruePeakDetector(version=2,oversamplingFactor=oversampling_factor)(audio_data)
  
  audio_data_length = len(audio_data)
  print("Audio data length:", audio_data_length)

  num_peaks = len(peak_locations) / oversampling_factor
  print("Number of peaks:", num_peaks)

  clipping_percentage = num_peaks / audio_data_length
  print("Clipping percentage:", clipping_percentage)

  return clipping_percentage

def noise_burst_percentage(audio_data, sample_rate):
  noiseBurstDetector = NoiseBurstDetector()

  noise_indexes = []
  for frame in FrameGenerator(audio_data, frameSize=512, hopSize=256):
    indexes = noiseBurstDetector(frame)
    noise_indexes.extend(indexes)

  # print ("Noise indexes:", noise_indexes)

  noise_burst_percentage = len(noise_indexes) / len(audio_data)
  print("Noise burst percentage:", noise_burst_percentage)

  return noise_burst_percentage

def compressibility_percentage(data):
    # Compress the data using zlib at the default compression level.

    # encode data as utf-8
    data = data.tobytes()

    compressed_data = zlib.compress(data)
    
    # Calculate the compression ratio.
    compression_ratio = len(compressed_data) / len(data)
    print(f'Compression ratio: {compression_ratio:.2f}')
    
    # Calculate the percentage of size reduction.
    compressibility = (1 - compression_ratio)
    print(f'Compressibility: {compressibility:.2f}%')
    
    return compressibility