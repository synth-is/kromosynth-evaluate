# sound quality based on Sound quality (SQ) metrics
# - utilising https://github.com/Eomys/MoSQITo

from mosqito.sq_metrics import roughness_dw, loudness_zwtv, equal_loudness_contours, sharpness_din_tv
import numpy as np

def roughness_dw_score_average(audio_data, sample_rate):
  r, r_spec, bark, time = roughness_dw(audio_data, sample_rate, overlap=0)
  # average the roughness values from r
  roughness_single_value = sum(r) / len(r)
  print('Roughness single value:', roughness_single_value)
  return roughness_single_value

def roughness_dw_score_median(audio_data, sample_rate):
  r, r_spec, bark, time = roughness_dw(audio_data, sample_rate, overlap=0)
  # median the roughness values from r
  roughness_single_value = np.median(r)
  print('Roughness single value:', roughness_single_value)
  return roughness_single_value

def roughness_dw_score_95th_percentile(audio_data, sample_rate):
  r, r_spec, bark, time = roughness_dw(audio_data, sample_rate, overlap=0)
  # 95th percentile the roughness values from r
  roughness_single_value = np.percentile(r, 95)
  print('Roughness single value:', roughness_single_value)
  return roughness_single_value

# "field_type", that can be set to "free" or "diffuse" depending on the environment of the audio signal recording: https://github.com/Eomys/MoSQITo/blob/master/tutorials/tuto_loudness_zwtv.ipynb
def loudness_zwicker_score_average(audio_data, sample_rate):
  l, l_spec, bark, time = loudness_zwtv(audio_data, sample_rate, field_type="free")
  # average the loudness values from l
  loudness_single_value = sum(l) / len(l)
  print('Loudness single value:', loudness_single_value)
  return loudness_single_value

def loudness_zwicker_score_median(audio_data, sample_rate):
  l, l_spec, bark, time = loudness_zwtv(audio_data, sample_rate, field_type="free")
  # median the loudness values from l
  loudness_single_value = np.median(l)
  print('Loudness single value:', loudness_single_value)
  return loudness_single_value

def loudness_zwicker_score_95th_percentile(audio_data, sample_rate):
  l, l_spec, bark, time = loudness_zwtv(audio_data, sample_rate, field_type="free")
  # 95th percentile the loudness values from l
  loudness_single_value = np.percentile(l, 95)
  print('Loudness single value:', loudness_single_value)
  return loudness_single_value

def equal_loudness_contour_score_average(audio_data, sample_rate):
  e, e_spec, bark, time = equal_loudness_contours(audio_data, sample_rate, overlap=0)
  # average the equal loudness contour values from e
  equal_loudness_contour_single_value = sum(e) / len(e)
  print('Equal loudness contour single value:', equal_loudness_contour_single_value)
  return equal_loudness_contour_single_value

def equal_loudness_contour_score_median(audio_data, sample_rate):
  e, e_spec, bark, time = equal_loudness_contours(audio_data, sample_rate, overlap=0)
  # median the equal loudness contour values from e
  equal_loudness_contour_single_value = np.median(e)
  print('Equal loudness contour single value:', equal_loudness_contour_single_value)
  return equal_loudness_contour_single_value

def equal_loudness_contour_score_95th_percentile(audio_data, sample_rate):
  e, e_spec, bark, time = equal_loudness_contours(audio_data, sample_rate, overlap=0)
  # 95th percentile the equal loudness contour values from e
  equal_loudness_contour_single_value = np.percentile(e, 95)
  print('Equal loudness contour single value:', equal_loudness_contour_single_value)
  return equal_loudness_contour_single_value