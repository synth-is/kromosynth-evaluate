# Quality Musicality Evaluation Service

A configurable WebSocket service for evaluating musical quality of synthesized sounds, designed for integration with Quality Diversity evolutionary sound synthesis.

## Quick Start

### 1. Start the Service

```bash
cd /Users/bjornpjo/Developer/apps/kromosynth-evaluate/evaluation/unsupervised
python quality_musicality.py --port 32051 --sample-rate 16000
```

### 2. Available Configuration Presets

Connect to the service using different endpoints:

**Noise Filtering Only (Phase 1 - Fastest)**
```
ws://127.0.0.1:32051/musicality?config_preset=noise_only
```
- SNR check (threshold: 6dB)
- Spectral entropy check (max: 0.95)
- Attack detection
- Spectral peaks detection

**Spectral Clarity (Phase 2 - Balanced)**
```
ws://127.0.0.1:32051/musicality?config_preset=spectral_clarity
```
- All noise filters
- Spectral concentration
- Harmonic-to-noise ratio (HNR)
- Spectral stability
- Harmonicity measurement

**VI-Focused (Phase 3 - Multi-Pitch)**
```
ws://127.0.0.1:32051/musicality?config_preset=vi_focused
```
- Stricter noise filters (SNR > 8dB)
- Full spectral clarity
- VI coherence evaluation (requires multi-pitch data)

### 3. Custom Configuration via URL

Override specific parameters:
```
ws://127.0.0.1:32051/musicality?noise_filters.snr_threshold=8.0&spectral_clarity.enabled=true
```

## Integration with QD Search

### Step 1: Update evolution-run-config.jsonc

```jsonc
{
  "evaluationQualityServers": ["ws://127.0.0.1:32051"],
  "classConfigurations": [
    {
      "refSetName": "musicalityFiltered",
      "featureExtractionEndpoint": "/mfcc-sans0-statistics",
      "featureExtractionType": "mfcc-sans0-statistics",
      "qualityEvaluationEndpoint": "/musicality?config_preset=noise_only",
      "qualityFromFeatures": true,
      "projectionEndpoint": "/pca",
      "shouldRetrainProjection": true,
      "sampleRate": 16000
    }
  ]
}
```

### Step 2: Test with Existing QD Run

```bash
cd /Users/bjornpjo/Developer/apps/kromosynth-cli/cli-app
npm run quality-diversity-search -- \
  --evolution-run-config-json-file conf/evolution-runs_quality-musicality.jsonc
```

## Evaluation Flow

### Single-Pitch Evaluation (Current)

```
QD Search → Render genome → Send audio buffer →
Musicality Service → Evaluate quality → Return fitness score
```

### Multi-Pitch Evaluation (Phase 3 - Future)

```
QD Search → Render genome → Send audio buffer →
Musicality Service → Fast checks → Return "requires_multi_pitch_evaluation" →
QD Search → Render at [-12, 0, +12] semitones → Send all buffers →
Musicality Service → Full VI evaluation → Return VI coherence scores
```

## Response Format

### Successful Evaluation

```json
{
  "fitness": 0.78,
  "rejected": false,
  "scores": {
    "noise_snr": 0.85,
    "noise_spectral_entropy": 0.72,
    "noise_attack": 1.0,
    "noise_spectral_peaks": 1.0,
    "clarity_concentration": 0.68,
    "clarity_hnr": 0.74,
    "clarity_stability": 0.81,
    "clarity_harmonicity": 0.70
  },
  "sound_type": "one-shot",
  "sound_type_confidence": 0.8,
  "evaluation_time": 0.043
}
```

### Rejection

```json
{
  "fitness": 0.0,
  "rejected": true,
  "reason": "failed_snr",
  "metadata": {
    "snr": 4.2,
    "threshold": 6.0
  },
  "evaluation_time": 0.008
}
```

### Multi-Pitch Required (Phase 3)

```json
{
  "fitness": null,
  "requires_multi_pitch_evaluation": true,
  "test_pitches": [-12, 0, 12],
  "preliminary_scores": {
    "noise_snr": 0.92,
    "clarity_hnr": 0.81
  },
  "evaluation_time": 0.035
}
```

## Testing the Service

### 1. Create Test Audio

```python
import numpy as np
import websocket

# Generate test tone
sample_rate = 16000
duration = 2.0
freq = 440.0
t = np.linspace(0, duration, int(sample_rate * duration))
audio = (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)

# Send to service
ws = websocket.create_connection("ws://127.0.0.1:32051/musicality?config_preset=noise_only")
ws.send_binary(audio.tobytes())
result = ws.recv()
print(result)
ws.close()
```

### 2. Test Rejection

```python
# Generate white noise
noise = np.random.randn(sample_rate * 2).astype(np.float32) * 0.1

ws = websocket.create_connection("ws://127.0.0.1:32051/musicality?config_preset=noise_only")
ws.send_binary(noise.tobytes())
result = ws.recv()
print(result)  # Should be rejected for high entropy
ws.close()
```

## Configuration Reference

### Noise Filters Config

```python
"noise_filters": {
    "enabled": True,
    "check_snr": True,
    "snr_threshold": 6.0,          # Minimum SNR in dB
    "snr_weight": 0.3,
    "check_spectral_entropy": True,
    "spectral_entropy_max": 0.95,   # Max entropy (0-1)
    "spectral_entropy_weight": 0.25,
    "require_attack": True,         # Must have clear attack
    "attack_weight": 0.25,
    "require_spectral_peaks": True, # Must have tonal content
    "spectral_peaks_weight": 0.2
}
```

### Spectral Clarity Config

```python
"spectral_clarity": {
    "enabled": True,
    "weight": 0.5,                  # Overall weight in final score
    "measure_concentration": True,
    "concentration_weight": 0.3,
    "measure_hnr": True,
    "hnr_weight": 0.3,
    "measure_stability": True,
    "stability_weight": 0.2,
    "measure_harmonicity": True,
    "harmonicity_weight": 0.2
}
```

### VI Coherence Config (Phase 3)

```python
"vi_coherence": {
    "enabled": True,
    "weight": 0.5,
    "test_pitches": [-12, 0, 12],   # Semitone deltas to test
    "measure_pitch_coherence": True,
    "coherence_weight": 0.5,
    "measure_attack_consistency": True,
    "attack_consistency_weight": 0.3,
    "min_snr_for_vi": 8.0,          # Higher threshold for VI
    "min_clarity_for_vi": 0.6
}
```

## Performance Notes

- **noise_only preset**: ~5-10ms per evaluation
- **spectral_clarity preset**: ~30-50ms per evaluation
- **vi_focused preset**: ~100-200ms per evaluation (with multi-pitch)

## Tuning Recommendations

1. **Start Conservative**: Use strict thresholds, then relax based on results
2. **Monitor Rejection Rate**: Target 30-50% rejection initially
3. **False Positive Target**: Keep below 5% (good sounds rejected)
4. **Listen & Iterate**: Manual validation is crucial

## Next Steps

1. Test standalone service with various audio types
2. Integrate with QD search and run comparison runs
3. Tune thresholds based on rejection rates
4. Listen to samples of accepted/rejected sounds
5. Adjust weights and add Phase 2 metrics if needed
6. Consider Phase 3 (VI coherence) only if generating VIworthy sounds
