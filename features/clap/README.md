# CLAP Feature Extraction

CLAP (Contrastive Language-Audio Pretraining) feature extractor for kromosynth QD pipeline.

## Overview

This module provides 512-dimensional audio embeddings using the LAION-CLAP model. These embeddings capture perceptual audio characteristics and can be used as:

- Input for learned behavior descriptors (QDHF projection)
- Direct features for quality diversity search
- Similarity metrics for sound recommendation

## Components

- **`clap_extractor.py`**: CLAPExtractor class for embedding extraction
- **`ws_clap_service.py`**: WebSocket service for remote extraction
- **`start_clap_service.sh`**: Service startup script

## Installation

Dependencies are already included in `kromosynth-evaluate/evaluation/unsupervised/requirements.txt`:

```txt
laion-clap==1.1.7
torch>=2.0.0
```

If not installed, run:

```bash
pip install laion-clap torch
```

## Quick Start

### Python API

```python
from features.clap.clap_extractor import CLAPExtractor
import numpy as np

# Initialize extractor
extractor = CLAPExtractor(device='cuda')  # or 'cpu'

# Generate or load audio (float32, mono)
audio = np.random.randn(16000).astype(np.float32)  # 1 second at 16kHz

# Extract embedding
embedding = extractor.extract_embedding(audio, sample_rate=16000)
print(embedding.shape)  # (512,)

# Batch extraction
audio_list = [audio1, audio2, audio3]
embeddings = extractor.extract_batch(audio_list, sample_rate=16000)
print(embeddings.shape)  # (3, 512)

# Compute similarity
similarity = extractor.compute_similarity(embedding1, embedding2)
print(f"Similarity: {similarity:.4f}")  # 0.0 to 1.0
```

### WebSocket Service

Start the service:

```bash
# Using the startup script
./features/clap/start_clap_service.sh

# Or with custom options
./features/clap/start_clap_service.sh --port 32051 --device cuda

# Or directly with Python
python -m features.clap.ws_clap_service --port 32051 --device cpu
```

Client example (JavaScript/Node.js):

```javascript
const WebSocket = require('websockets');

const ws = new WebSocket('ws://localhost:32051/clap');

// Send audio buffer (binary)
const audioBuffer = new Float32Array(16000);  // Your audio data
ws.send(audioBuffer.buffer);

// Or send JSON with metadata
ws.send(JSON.stringify({
    audio_buffer: Buffer.from(audioBuffer.buffer).toString('base64'),
    sample_rate: 16000
}));

// Receive embedding
ws.on('message', (data) => {
    const response = JSON.parse(data);
    console.log('Embedding:', response.embedding);  // Array of 512 floats
    console.log('Time:', response.extraction_time_ms, 'ms');
});
```

## Configuration

### Environment Variables

- `CLAP_CHECKPOINT_PATH`: Path to CLAP checkpoint file (optional)
- `CLAP_DEVICE`: Device to use (`cuda` or `cpu`)

### Checkpoint Download

On first run, the default CLAP checkpoint will be downloaded automatically (~500MB):
- Model: `music_audioset_epoch_15_esc_90.14.pt`
- Trained on: AudioSet music subset
- Performance: 90.14% on ESC-50 classification

The checkpoint is cached in `~/.cache/laion-clap/` by default.

To use a custom checkpoint:

```python
extractor = CLAPExtractor(checkpoint_path='/path/to/checkpoint.pt')
```

Or via environment variable:

```bash
export CLAP_CHECKPOINT_PATH=/path/to/checkpoint.pt
./features/clap/start_clap_service.sh
```

## Testing

### Import Test

```bash
python -c "from features.clap import CLAPExtractor; print('OK')"
```

### Simple Functional Test

```bash
python scripts/test_clap_simple.py
```

This will:
1. Initialize the extractor (downloads checkpoint on first run)
2. Test single and batch extraction
3. Verify embedding consistency
4. Test edge cases (silent audio, resampling)
5. Benchmark extraction speed

Expected output:
```
[1/7] Initializing CLAP extractor...
✓ Extractor initialized in 2.34s
[2/7] Testing single embedding extraction...
✓ Extracted embedding in 45.23ms
...
All tests passed! ✓
```

### Unit Tests (with pytest)

```bash
pytest test/test_clap_extractor.py -v
```

## Performance

### Latency (CPU - Apple M2)

- Single extraction: ~40-60ms
- Batch (10 sounds): ~200-300ms (~20-30ms per sound)

### Latency (GPU - NVIDIA RTX 3090)

- Single extraction: ~10-15ms
- Batch (10 sounds): ~50-80ms (~5-8ms per sound)

**Target:** < 100ms per sound for real-time QD loop

### Memory Usage

- Model size: ~80MB (HTSAT-tiny architecture)
- Per-audio overhead: ~5MB
- Batch processing recommended for efficiency

## Audio Preprocessing

The extractor automatically handles:

1. **Resampling**: Input audio is resampled to 48kHz (CLAP requirement)
2. **Mono conversion**: Stereo audio is converted to mono
3. **Normalization**: Audio is normalized to [-1, 1] range

Supported input formats:
- Sample rates: Any (will be resampled to 48kHz)
- Channels: Mono or stereo (stereo averaged to mono)
- Data type: float32 (recommended) or any numeric type

## Integration with kromosynth-cli

Add to evolution configuration:

```jsonc
{
  "featureExtractionEndpoint": "/clap",
  "featureExtractionServers": ["ws://127.0.0.1:32051"]
}
```

The kromosynth-cli will:
1. Render genome to audio
2. Send audio to CLAP service
3. Receive 512D embedding
4. Use embedding for BD projection (via QDHF) or quality evaluation

## Next Steps (Phase 2-3)

1. **QDHF Projection**: Train learned projection from 512D CLAP → 6D behavior descriptors
2. **Combined Endpoint**: Create `/clap-bd` endpoint that returns both CLAP embedding and projected BD
3. **Integration**: Connect with pyribs QD service for CMA-MAE search

## Troubleshooting

### Checkpoint download fails

If the automatic download fails, manually download:

```bash
mkdir -p ~/.cache/laion-clap
cd ~/.cache/laion-clap
wget https://huggingface.co/laion/clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt
```

Then use:
```bash
export CLAP_CHECKPOINT_PATH=~/.cache/laion-clap/music_audioset_epoch_15_esc_90.14.pt
```

### CUDA out of memory

Reduce batch size or use CPU:

```bash
./features/clap/start_clap_service.sh --device cpu
```

### Slow extraction on CPU

Expected on CPU. For production:
- Use GPU (10x faster)
- Batch sounds together
- Consider caching embeddings for known sounds

## References

- **LAION-CLAP**: https://github.com/LAION-AI/CLAP
- **Paper**: [Large-scale Contrastive Language-Audio Pretraining with Feature Fusion and Keyword-to-Caption Augmentation](https://arxiv.org/abs/2211.06687)
- **Model Card**: https://huggingface.co/laion/clap

## License

CLAP model: MIT License (LAION-AI)
This integration: Same as kromosynth-evaluate
