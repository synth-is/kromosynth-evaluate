"""
Spectral clarity and quality measurements for musical sounds.

Uses librosa for all audio analysis (no Essentia dependency).
"""

import numpy as np
import librosa


def spectral_concentration_score(audio_data, sample_rate, frame_size=2048, hop_length=512):
    """
    Measure how concentrated energy is in specific frequency bands.
    
    High concentration = clear tonal content
    Low concentration = noisy/diffuse spectrum
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        frame_size: Size of FFT frame
        hop_length: Hop length
        
    Returns:
        float: Concentration score (0-1, higher = more concentrated)
    """
    # Compute STFT
    D = librosa.stft(audio_data, n_fft=frame_size, hop_length=hop_length)
    magnitude = np.abs(D)
    
    concentration_scores = []
    
    for frame_idx in range(magnitude.shape[1]):
        frame_spec = magnitude[:, frame_idx]
        
        # Calculate how much energy is in top 10% of bins
        sorted_magnitudes = np.sort(frame_spec)[::-1]
        total_energy = np.sum(frame_spec)
        
        if total_energy > 0:
            top_10_percent = int(len(frame_spec) * 0.1)
            concentrated_energy = np.sum(sorted_magnitudes[:top_10_percent])
            concentration = concentrated_energy / total_energy
            concentration_scores.append(concentration)
    
    result = np.mean(concentration_scores) if concentration_scores else 0.0
    return float(result)  # Convert numpy float to Python float for JSON serialization


def harmonic_to_noise_ratio(audio_data, sample_rate, frame_size=2048, hop_length=512):
    """
    Estimate the ratio of harmonic to noise content.
    
    Higher values indicate clearer tonal/harmonic content.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        frame_size: Size of FFT frame
        hop_length: Hop length
        
    Returns:
        float: HNR score (0-1, higher = more harmonic)
    """
    # Use harmonic-percussive source separation
    y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
    
    # Calculate energy in each component
    harmonic_energy = np.sum(y_harmonic ** 2)
    total_energy = np.sum(audio_data ** 2)
    
    if total_energy == 0:
        return 0.0
    
    hnr = harmonic_energy / total_energy
    return float(min(1.0, hnr))  # Convert to Python float


def spectral_stability_score(audio_data, sample_rate, frame_size=2048, hop_length=512):
    """
    Measure how stable the spectral content is over time.
    
    Stable spectra are generally more musical.
    High variability may indicate noise or instability.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        frame_size: Size of FFT frame
        hop_length: Hop length
        
    Returns:
        float: Stability score (0-1, higher = more stable)
    """
    # Compute spectral centroid over time
    centroids = librosa.feature.spectral_centroid(
        y=audio_data, 
        sr=sample_rate,
        n_fft=frame_size,
        hop_length=hop_length
    )[0]
    
    if len(centroids) < 2:
        return 0.0
    
    # Calculate coefficient of variation (std/mean)
    mean_centroid = np.mean(centroids)
    std_centroid = np.std(centroids)
    
    if mean_centroid > 0:
        cv = std_centroid / mean_centroid
        # Normalize: high CV (>1) = unstable, low CV (<0.1) = stable
        stability = max(0.0, min(1.0, 1.0 - cv))
        return float(stability)  # Convert to Python float
    
    return 0.0


def inharmonicity_score(audio_data, sample_rate, frame_size=2048):
    """
    Measure inharmonicity of the sound.
    
    Low inharmonicity = clear harmonic structure
    High inharmonicity = noisy or metallic character
    
    Uses spectral flatness as a proxy for inharmonicity.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        frame_size: Size of FFT frame
        
    Returns:
        float: Harmonicity score (0-1, higher = more harmonic)
    """
    # Compute spectral flatness (inverse of harmonicity)
    flatness = librosa.feature.spectral_flatness(
        y=audio_data,
        n_fft=frame_size
    )[0]
    
    if len(flatness) == 0:
        return 0.5
    
    avg_flatness = np.mean(flatness)
    
    # Convert flatness to harmonicity
    # Flatness near 0 = harmonic, near 1 = noisy
    harmonicity = 1.0 - avg_flatness
    
    return float(max(0.0, min(1.0, harmonicity)))  # Convert to Python float


def calculate_spectral_clarity_scores(audio_data, sample_rate, config):
    """
    Calculate all spectral clarity scores based on configuration.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        config: Dictionary with spectral clarity settings
        
    Returns:
        dict: Scores for each enabled metric (all Python floats for JSON serialization)
    """
    scores = {}
    
    if config.get("measure_concentration", True):
        scores["concentration"] = spectral_concentration_score(audio_data, sample_rate)
    
    if config.get("measure_hnr", True):
        scores["hnr"] = harmonic_to_noise_ratio(audio_data, sample_rate)
    
    if config.get("measure_stability", True):
        scores["stability"] = spectral_stability_score(audio_data, sample_rate)
    
    if config.get("measure_harmonicity", True):
        scores["harmonicity"] = inharmonicity_score(audio_data, sample_rate)
    
    return scores
