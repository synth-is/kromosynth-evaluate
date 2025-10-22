# measurements/quality/vi_coherence.py

import numpy as np
import librosa
from typing import Dict, Tuple

def requires_multi_pitch_evaluation(single_pitch_scores: Dict, config: Dict) -> bool:
    """
    Determine if this sound warrants multi-pitch VI coherence evaluation.
    
    Only evaluate VI coherence for sounds that:
    - Passed noise filters
    - Have good spectral clarity
    - Show harmonic character
    
    Args:
        single_pitch_scores: Scores from Phase 1 single-pitch evaluation
        config: VI coherence configuration with thresholds
        
    Returns:
        bool: True if this sound should be evaluated for VI coherence
    """
    if not config.get("enabled", False):
        return False
    
    # Strict thresholds - only test promising sounds
    min_snr = config.get("min_snr_for_vi", 8.0)
    min_hnr = config.get("min_hnr_for_vi", 0.6)
    min_harmonicity = config.get("min_harmonicity_for_vi", 0.5)
    
    # Check scores from Phase 1
    snr_score = single_pitch_scores.get("noise_snr", 0)
    hnr_score = single_pitch_scores.get("clarity_hnr", 0)
    harmonicity_score = single_pitch_scores.get("clarity_harmonicity", 0)
    
    # Convert normalized SNR score back to raw dB value for comparison
    # SNR was normalized by dividing by 20
    snr_raw = snr_score * 20.0

    passed_snr = snr_raw >= min_snr
    passed_hnr = hnr_score >= min_hnr
    passed_harmonicity = harmonicity_score >= min_harmonicity

    passed = passed_snr and passed_hnr and passed_harmonicity

    print(
        f"VI coherence decision: snr_raw={snr_raw:.2f} (th={min_snr:.2f}) -> {passed_snr}; "
        f"hnr={hnr_score:.3f} (th={min_hnr:.3f}) -> {passed_hnr}; "
        f"harmonicity={harmonicity_score:.3f} (th={min_harmonicity:.3f}) -> {passed_harmonicity}; "
        f"passed={passed}"
    )
    
    return passed


def multi_pitch_coherence_score(
    audio_buffers_by_pitch: Dict[int, np.ndarray], 
    sample_rate: int
) -> float:
    """
    Measure spectral similarity across pitch transpositions.
    
    This evaluates whether a sound maintains consistent timbral character
    across different pitches - a key requirement for VI-worthiness.
    
    Args:
        audio_buffers_by_pitch: Dict like {-12: array1, 0: array2, 12: array3}
        sample_rate: Audio sample rate
    
    Returns:
        float: Coherence score (0-1, higher = more consistent across pitches)
    """
    # Extract spectral features for each pitch
    spectral_features = {}
    
    for pitch_delta, audio in audio_buffers_by_pitch.items():
        # Use spectral centroid and rolloff as timbre descriptors
        centroid = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)
        bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)
        
        # Normalize and combine
        features = np.concatenate([
            centroid.flatten() / (sample_rate / 2),
            rolloff.flatten() / (sample_rate / 2),
            bandwidth.flatten() / (sample_rate / 2)
        ])
        
        spectral_features[pitch_delta] = features
    
    # Calculate pairwise similarities between adjacent octaves
    pitches = sorted(audio_buffers_by_pitch.keys())
    similarities = []
    
    for i in range(len(pitches) - 1):
        pitch1 = pitches[i]
        pitch2 = pitches[i + 1]
        
        feat1 = spectral_features[pitch1]
        feat2 = spectral_features[pitch2]
        
        # Ensure same length
        min_len = min(len(feat1), len(feat2))
        feat1 = feat1[:min_len]
        feat2 = feat2[:min_len]
        
        # Cosine similarity
        dot_product = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            similarities.append(similarity)
    
    # Average similarity across adjacent octaves
    if not similarities:
        return 0.5  # Neutral score if no comparisons possible
    
    coherence = float(np.mean(similarities))
    return max(0.0, min(1.0, coherence))


def attack_consistency_score(
    audio_buffers_by_pitch: Dict[int, np.ndarray],
    sample_rate: int
) -> float:
    """
    Measure consistency of attack characteristics across pitches.
    
    VI sounds should have similar attack envelopes regardless of pitch.
    
    Args:
        audio_buffers_by_pitch: Dict of pitch delta to audio buffer
        sample_rate: Audio sample rate
        
    Returns:
        float: Consistency score (0-1, higher = more consistent attacks)
    """
    # Extract onset strength for each pitch
    attack_profiles = {}
    
    for pitch_delta, audio in audio_buffers_by_pitch.items():
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        
        # Normalize
        if len(onset_env) > 0 and np.max(onset_env) > 0:
            onset_env = onset_env / (np.max(onset_env) + 1e-6)
            attack_profiles[pitch_delta] = onset_env
    
    if not attack_profiles:
        return 0.5
    
    # Compare attack shapes between adjacent pitches
    pitches = sorted(attack_profiles.keys())
    consistencies = []
    
    for i in range(len(pitches) - 1):
        env1 = attack_profiles[pitches[i]]
        env2 = attack_profiles[pitches[i + 1]]
        
        # Compare first 10% of envelopes (attack portion)
        attack_len = min(len(env1), len(env2)) // 10
        if attack_len < 2:
            continue
            
        attack1 = env1[:attack_len]
        attack2 = env2[:attack_len]
        
        # Correlation of attack shapes
        if len(attack1) > 1 and len(attack2) > 1:
            correlation = np.corrcoef(attack1, attack2)[0, 1]
            if not np.isnan(correlation):
                consistencies.append(correlation)
    
    if not consistencies:
        return 0.5
    
    consistency = float(np.mean(consistencies))
    return max(0.0, min(1.0, consistency))


def spectral_stability_score(
    audio_buffers_by_pitch: Dict[int, np.ndarray],
    sample_rate: int
) -> float:
    """
    Measure how stable the spectral envelope shape is across pitches.
    
    Good VI sounds maintain their character (formants, spectral shape)
    even as harmonics shift with pitch.
    
    Args:
        audio_buffers_by_pitch: Dict of pitch delta to audio buffer
        sample_rate: Audio sample rate
        
    Returns:
        float: Stability score (0-1, higher = more stable spectral character)
    """
    # Extract MFCC features (capture spectral envelope)
    mfcc_features = {}
    
    for pitch_delta, audio in audio_buffers_by_pitch.items():
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        # Average over time to get overall spectral character
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_features[pitch_delta] = mfcc_mean
    
    # Compare spectral shapes across pitches
    pitches = sorted(mfcc_features.keys())
    similarities = []
    
    for i in range(len(pitches) - 1):
        mfcc1 = mfcc_features[pitches[i]]
        mfcc2 = mfcc_features[pitches[i + 1]]
        
        # Cosine similarity of MFCC profiles
        dot_product = np.dot(mfcc1, mfcc2)
        norm1 = np.linalg.norm(mfcc1)
        norm2 = np.linalg.norm(mfcc2)
        
        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
            similarities.append(similarity)
    
    if not similarities:
        return 0.5
    
    stability = float(np.mean(similarities))
    return max(0.0, min(1.0, stability))


def evaluate_vi_coherence(
    audio_buffers_by_pitch: Dict[int, np.ndarray],
    sample_rate: int,
    config: Dict
) -> Dict:
    """
    Complete VI coherence evaluation combining multiple metrics.
    
    Args:
        audio_buffers_by_pitch: Audio renders at different pitches
        sample_rate: Sample rate
        config: VI coherence configuration with weights
        
    Returns:
        Dict with individual scores and weighted overall score
    """
    # Calculate individual metrics
    pitch_coherence = multi_pitch_coherence_score(audio_buffers_by_pitch, sample_rate)
    attack_consistency = attack_consistency_score(audio_buffers_by_pitch, sample_rate)
    spectral_stability = spectral_stability_score(audio_buffers_by_pitch, sample_rate)
    
    # Get weights from config
    weights = {
        'pitch_coherence': config.get('pitch_coherence_weight', 0.5),
        'attack_consistency': config.get('attack_consistency_weight', 0.3),
        'spectral_stability': config.get('spectral_stability_weight', 0.2)
    }
    
    # Weighted combination
    overall_score = (
        pitch_coherence * weights['pitch_coherence'] +
        attack_consistency * weights['attack_consistency'] +
        spectral_stability * weights['spectral_stability']
    )
    
    # Determine sound type and confidence
    sound_type = "vi_worthy" if overall_score > 0.7 else "one_shot"
    confidence = overall_score if sound_type == "vi_worthy" else (1.0 - overall_score)
    
    return {
        'vi_pitch_coherence': pitch_coherence,
        'vi_attack_consistency': attack_consistency,
        'vi_spectral_stability': spectral_stability,
        'vi_overall_score': overall_score,
        'sound_type': sound_type,
        'sound_type_confidence': confidence
    }


# Alias for backward compatibility with quality_musicality.py
calculate_vi_coherence_scores = evaluate_vi_coherence
