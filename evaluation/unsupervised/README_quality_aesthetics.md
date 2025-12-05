# Audiobox Aesthetics Evaluation Service

WebSocket service that evaluates audio using Meta's `audiobox-aesthetics` model.

## Overview

This service provides audio aesthetic evaluation across 4 dimensions:

- **CE** (Content Enjoyment) - How enjoyable the audio content is
- **CU** (Content Usefulness) - How useful/functional the audio content is
- **PC** (Production Complexity) - How complex the audio production is
- **PQ** (Production Quality) - How high-quality the audio production is

Each dimension returns a score (typically 0-10 range).

## Installation

First, install the audiobox-aesthetics library:

```bash
pip install audiobox_aesthetics
```

The service will automatically download the model from HuggingFace on first use (`facebook/audiobox-aesthetics`).

## Usage

### Quick Start

```bash
# Start with defaults (port 8085, 48kHz sample rate)
./start_quality_aesthetics.sh

# Specify custom port
./start_quality_aesthetics.sh 8090

# Specify custom sample rate
./start_quality_aesthetics.sh 8090 44100

# Use custom checkpoint file
./start_quality_aesthetics.sh 8090 48000 /path/to/checkpoint.pth
```

### Manual Start

```bash
python quality_aesthetics.py \
    --port 8085 \
    --sample-rate 48000 \
    --checkpoint-path /path/to/checkpoint.pth  # Optional
```

### Command Line Arguments

- `--host` - Host to run server on (default: `localhost`)
- `--port` - Port number (default: `8080`)
- `--sample-rate` - Audio sample rate in Hz (default: `48000`)
- `--checkpoint-path` - Path to model checkpoint (optional, uses HuggingFace default if not specified)
- `--process-title` - Process title for monitoring (default: `quality_aesthetics`)
- `--host-info-file` - Host info file for HPC environments

## WebSocket API

### Endpoint

```
ws://localhost:8085/?output_mode=all
```

### Query Parameters

- `output_mode` - Output format:
  - `all` (default) - Returns all 4 dimension scores
  - `top` - Returns only the highest scoring dimension

### Request Format

Send audio data as **binary message** (Float32Array):

```javascript
// JavaScript example
const audioBuffer = new Float32Array([...]); // Your audio samples
websocket.send(audioBuffer);
```

### Response Formats

#### Mode: `all` (Default)

Returns all dimension scores:

```json
{
  "status": "received standalone audio",
  "taggedPredictions": {
    "CE_ContentEnjoyment": 5.146,
    "CU_ContentUsefulness": 5.779,
    "PC_ProductionComplexity": 2.148,
    "PQ_ProductionQuality": 7.220
  }
}
```

#### Mode: `top`

Returns only the highest scoring dimension:

```json
{
  "status": "received standalone audio",
  "fitness": {
    "top_score": 7.220,
    "index": 3,
    "top_score_class": "ProductionQuality"
  }
}
```

## Integration Examples

### Node.js Client

```javascript
const WebSocket = require('ws');

const ws = new WebSocket('ws://localhost:8085/?output_mode=all');

ws.on('open', () => {
  // Send audio buffer (Float32Array)
  const audioData = new Float32Array([/* your audio samples */]);
  ws.send(audioData);
});

ws.on('message', (data) => {
  const result = JSON.parse(data);
  console.log('Aesthetics scores:', result.taggedPredictions);
});
```

### Python Client

```python
import websocket
import numpy as np
import json

ws = websocket.create_connection('ws://localhost:8085/?output_mode=top')

# Send audio buffer
audio_data = np.array([...], dtype=np.float32)
ws.send(audio_data.tobytes())

# Receive result
result = json.loads(ws.recv())
print(f"Top score: {result['fitness']['top_score']}")
print(f"Dimension: {result['fitness']['top_score_class']}")

ws.close()
```

## Use in QD Evolution

### As Fitness Function

Use `output_mode=top` to get a single fitness score:

```javascript
// In your evolution configuration
{
  "fitnessFunction": "audiobox_aesthetics",
  "fitnessServiceUrl": "ws://localhost:8085/?output_mode=top"
}
```

### As Behavior Dimensions

Use `output_mode=all` and extract individual dimensions:

```javascript
// Get all 4 dimensions for QD behavior space
const result = await evaluateAudio(audioBuffer);
const dimensions = [
  result.taggedPredictions.CE_ContentEnjoyment,
  result.taggedPredictions.CU_ContentUsefulness,
  result.taggedPredictions.PC_ProductionComplexity,
  result.taggedPredictions.PQ_ProductionQuality
];
```

## Performance Notes

- **Model loading**: Takes a few seconds on first startup (downloads from HuggingFace)
- **Inference time**: Typically 0.5-2 seconds per audio sample (depends on length and GPU availability)
- **GPU acceleration**: Automatically uses GPU if available via PyTorch
- **Audio length**: Model accepts various audio lengths, but longer audio takes more time

## Troubleshooting

### Import Error

```
✗ Error: audiobox_aesthetics not installed
```

**Solution**: Install the library:
```bash
pip install audiobox_aesthetics
```

### Model Download Fails

If HuggingFace download fails, you can:
1. Check your internet connection
2. Download the checkpoint manually and use `--checkpoint-path`
3. Set HuggingFace cache directory: `export HF_HOME=/path/to/cache`

### Dimension Mismatch

Ensure your audio buffer:
- Is Float32 format
- Matches the specified `--sample-rate`
- Is mono or stereo (service handles both)

## References

- [audiobox-aesthetics GitHub](https://github.com/facebookresearch/audiobox-aesthetics)
- [HuggingFace Model](https://huggingface.co/facebook/audiobox-aesthetics)
- Model paper: *Audiobox Aesthetics: A Universal Audio Quality Estimator* (Meta AI)
