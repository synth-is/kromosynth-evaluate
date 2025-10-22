"""
Noise detection and filtering for musical sound quality assessment.

Uses librosa for all audio analysis (no Essentia dependency).
"""

import numpy as np
import librosa


def signal_to_noise_ratio(audio_data, sample_rate, frame_size=2048, hop_length=512):
    """
    Calculate the signal-to-noise ratio using spectral analysis.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        frame_size: Size of analysis frame
        hop_length: Hop length for frames
        
    Returns:
        float: SNR in dB (returns 0 if cannot calculate)
    """
    # Compute STFT
    D = librosa.stft(audio_data, n_fft=frame_size, hop_length=hop_length)
    magnitude = np.abs(D)
    
    # Estimate signal and noise
    # Signal: top 20% of spectral energy
    # Noise: bottom 20% of spectral energy
    sorted_mag = np.sort(magnitude.flatten())
    signal_threshold = np.percentile(sorted_mag, 80)
    noise_threshold = np.percentile(sorted_mag, 20)
    
    signal_power = np.mean(magnitude[magnitude > signal_threshold] ** 2)
    noise_power = np.mean(magnitude[magnitude < noise_threshold] ** 2)
    
    if noise_power == 0 or signal_power == 0:
        return 0.0
    
    snr = 10 * np.log10(signal_power / noise_power)
    
    # Handle edge cases
    if np.isinf(snr) or np.isnan(snr):
        return 0.0
        
    return max(0.0, snr)


def spectral_entropy(audio_data, sample_rate, frame_size=2048, hop_length=512):
    """
    Calculate spectral entropy - high values indicate white noise.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        frame_size: Size of FFT frame
        hop_length: Hop length for frames
        
    Returns:
        float: Average spectral entropy (0-1, where 1 = white noise)
    """
    # Compute STFT
    D = librosa.stft(audio_data, n_fft=frame_size, hop_length=hop_length)
    magnitude = np.abs(D)
    
    entropies = []
    
    for frame_idx in range(magnitude.shape[1]):
        frame_spec = magnitude[:, frame_idx]
        
        # Calculate normalized power spectrum
        power = frame_spec ** 2
        power_sum = np.sum(power)
        
        if power_sum > 0:
            prob = power / power_sum
            # Avoid log(0)
            prob = prob[prob > 1e-10]
            entropy = -np.sum(prob * np.log2(prob))
            # Normalize by maximum entropy
            max_entropy = np.log2(len(prob))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            entropies.append(normalized_entropy)
    
    return np.mean(entropies) if entropies else 1.0


def has_clear_attack(audio_data, sample_rate, threshold=0.01):
    """
    Check if sound has a clear attack transient.
    
    Uses onset detection to find attacks.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        threshold: Minimum onset strength
        
    Returns:
        bool: True if clear attack detected
    """
    # Compute onset strength envelope
    onset_env = librosa.onset.onset_strength(y=audio_data, sr=sample_rate)
    
    if len(onset_env) < 2:
        return False
    
    # Check for strong initial transient (first 10% of signal)
    window_size = max(1, len(onset_env) // 10)
    initial_max = np.max(onset_env[:window_size])
    overall_max = np.max(onset_env)
    
    # Attack should be in first portion and reach significant strength
    has_attack = initial_max > (overall_max * 0.5) and overall_max > threshold
    
    return has_attack


def has_spectral_peaks(audio_data, sample_rate, min_peaks=3, frame_size=2048):
    """
    Check if spectrum has distinct peaks (not flat/noisy).
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        min_peaks: Minimum number of spectral peaks required
        frame_size: Size of FFT frame
        
    Returns:
        bool: True if sufficient spectral peaks found
    """
    # Compute average spectrum
    D = librosa.stft(audio_data, n_fft=frame_size)
    avg_spectrum = np.mean(np.abs(D), axis=1)
    
    # Find peaks in spectrum
    # A peak is a local maximum that's significantly higher than neighbors
    peaks = []
    for i in range(1, len(avg_spectrum) - 1):
        if avg_spectrum[i] > avg_spectrum[i-1] and avg_spectrum[i] > avg_spectrum[i+1]:
            # Check if peak is significant (above median)
            if avg_spectrum[i] > np.median(avg_spectrum) * 1.5:
                peaks.append(i)
    
    return len(peaks) >= min_peaks


def has_periodicity(audio_data, sample_rate, min_correlation=0.1):
    """
    Check if sound has periodic structure (not random noise).
    
    Uses autocorrelation to detect repetitive patterns.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        min_correlation: Minimum autocorrelation value
        
    Returns:
        bool: True if periodic structure detected
    """
    # Analyze middle section (avoid silence at start/end)
    quarter_len = len(audio_data) // 4
    analysis_segment = audio_data[quarter_len:3*quarter_len]
    
    if len(analysis_segment) < sample_rate:  # Need at least 1 second
        return False
    
    # Calculate autocorrelation using librosa
    autocorr = librosa.autocorrelate(analysis_segment)
    
    # Normalize
    autocorr = autocorr / autocorr[0] if autocorr[0] != 0 else autocorr
    
    # Look for peaks in autocorrelation (excluding lag 0)
    if len(autocorr) < 2:
        return False
    
    # Find maximum correlation in reasonable pitch range
    # (20 Hz to 2000 Hz = 50ms to 0.5ms period)
    min_lag = int(sample_rate * 0.0005)  # 0.5ms (2000 Hz)
    max_lag = int(sample_rate * 0.05)     # 50ms (20 Hz)
    max_lag = min(max_lag, len(autocorr) - 1)
    
    if max_lag <= min_lag:
        return False
    
    max_correlation = np.max(autocorr[min_lag:max_lag])
    
    return max_correlation > min_correlation


def deterministic_rendering_score(audio_data):
    """
    Placeholder for deterministic rendering check.
    
    This requires rendering the same genome multiple times,
    which should be done at the QD search level, not here.
    
    For now, returns 1.0 (assume deterministic).
    
    Args:
        audio_data: Audio samples as numpy array
        
    Returns:
        float: Determinism score (0-1)
    """
    # TODO: Implement when QD search provides multiple renders
    return 1.0


def calculate_noise_filter_scores(audio_data, sample_rate, config):
    """
    Calculate all noise filter scores based on configuration.
    
    Args:
        audio_data: Audio samples as numpy array
        sample_rate: Sample rate in Hz
        config: Dictionary with noise filter settings
        
    Returns:
        dict: Scores for each enabled filter, or rejection reason
    """
    scores = {}
    
    # SNR check (fastest - do first)
    if config.get("check_snr", True):
        snr = signal_to_noise_ratio(audio_data, sample_rate)
        snr_threshold = config.get("snr_threshold", 6.0)
        
        if snr < snr_threshold:
            return {
                "rejected": True,
                "reason": "failed_snr",
                "snr": snr,
                "threshold": snr_threshold
            }
        
        # Normalize SNR to 0-1 score (20dB = excellent)
        scores["snr"] = min(1.0, snr / 20.0)
    
    # Spectral entropy check
    if config.get("check_spectral_entropy", True):
        entropy = spectral_entropy(audio_data, sample_rate)
        max_entropy = config.get("spectral_entropy_max", 0.95)
        
        if entropy > max_entropy:
            return {
                "rejected": True,
                "reason": "white_noise",
                "entropy": entropy,
                "threshold": max_entropy
            }
        
        scores["spectral_entropy"] = 1.0 - entropy
    
    # Attack check
    if config.get("require_attack", True):
        has_attack = has_clear_attack(audio_data, sample_rate)
        
        if not has_attack:
            return {
                "rejected": True,
                "reason": "no_attack"
            }
        
        scores["attack"] = 1.0
    
    # Spectral peaks check
    if config.get("require_spectral_peaks", True):
        has_peaks = has_spectral_peaks(audio_data, sample_rate)
        
        if not has_peaks:
            return {
                "rejected": True,
                "reason": "no_spectral_peaks"
            }
        
        scores["spectral_peaks"] = 1.0
    
    # Periodicity check
    if config.get("require_periodicity", False):
        has_period = has_periodicity(audio_data, sample_rate)
        
        if not has_period:
            return {
                "rejected": True,
                "reason": "no_periodicity"
            }
        
        scores["periodicity"] = 1.0
    
    return {
        "rejected": False,
        "scores": scores
    }
