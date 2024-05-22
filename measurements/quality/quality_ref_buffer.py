import os
import cdpam
import numpy as np
import torch
from .NFLossFunctions import STFT_loss

# TODO: this is incomplete and has issues, e.g. CDPAM a requires 2D tensor, not a 1D tensor, which seems to be supplied
# - and the interpretation / usage of STFT_loss is incomplete; see its useage in: https://github.com/RiccardoVib/Physics-Informed-Differentiable-Piano/blob/main/Code/Training.py

reference_audios = {}
def get_cdpam_distance(query_audio, reference_audio_path):
    global reference_audios
    if torch.backends.mps.is_available():
      loss_fn = cdpam.CDPAM(dev='mps')
    else:
      loss_fn = cdpam.CDPAM()
    if reference_audio_path is not None and os.path.exists(reference_audio_path) and not reference_audio_path in reference_audios:
        print(f"Loading reference audio from {reference_audio_path}")
        # load reference audio buffer from reference_audio_path
        # reference_audios[reference_audio_path] = np.frombuffer(open(reference_audio_path, 'rb').read(), dtype=np.float32)
        reference_audios[reference_audio_path] = cdpam.load_audio(reference_audio_path)
    reference_audio = reference_audios[reference_audio_path]
    dist = loss_fn.forward(reference_audio, query_audio)
    return dist

def get_multi_resolution_spectral_loss(query_audio, reference_audio_path):
    # - or Multi-resolution STFT loss
    # eq. 1 at https://www.duo.uio.no/bitstream/handle/10852/105127/DAFX23.pdf
    # further: 
    # - https://arxiv.org/pdf/1910.11480.pdf
    # - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10319088
    loss_metric = STFT_loss(m=[512, 1024, 2048])
    if reference_audio_path is not None and os.path.exists(reference_audio_path) and not reference_audio_path in reference_audios:
        print(f"Loading reference audio from {reference_audio_path}")
        # load reference audio buffer from reference_audio_path
        # reference_audios[reference_audio_path] = cdpam.load_audio(reference_audio_path)
        reference_audios[reference_audio_path] = np.frombuffer(open(reference_audio_path, 'rb').read(), dtype=np.float32)
    reference_audio = reference_audios[reference_audio_path]
    loss = loss_metric(reference_audio, query_audio)
    return loss