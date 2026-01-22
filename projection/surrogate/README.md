# Surrogate Quality Prediction Service

Predicts genome quality **before audio rendering** using structural features extracted from CPPN+DSP genomes. This enables selective evaluation in QD loops, significantly reducing compute costs.

## Architecture

- **Input**: 64D genome feature vector (from `GenomeFeatureExtractor`)
- **Model**: Deep ensemble of 5 MLPs for uncertainty quantification
- **Output**: Quality score [0,1] + epistemic uncertainty

## Quick Start

```bash
# Start service (fresh model)
./start_surrogate_service.sh

# Or with pre-trained model
./start_surrogate_service.sh --model models/surrogate/checkpoint.pt
```

Service runs on `ws://localhost:32070/predict` by default.

## API Usage

### Single Prediction (from genome)

```javascript
// Send
{
    "genome": { /* CPPN+DSP genome */ },
    "genome_id": "abc123"
}

// Receive
{
    "quality": 0.734,
    "uncertainty": 0.089,
    "genome_id": "abc123",
    "inference_time_ms": 2.34
}
```

### Single Prediction (from features)

```javascript
// Send
{
    "features": [/* 64 floats */],
    "genome_id": "abc123"
}

// Receive
{
    "quality": 0.734,
    "uncertainty": 0.089,
    "inference_time_ms": 1.12
}
```

### Batch Prediction

```javascript
// Send
{
    "genomes": [/* array of genomes */],
    "genome_ids": ["id1", "id2", ...]
}

// Receive
{
    "qualities": [0.734, 0.512, ...],
    "uncertainties": [0.089, 0.124, ...],
    "count": 50,
    "inference_time_ms": 15.6
}
```

### Online Training

```javascript
// Send
{
    "type": "train",
    "genomes": [/* genomes that were evaluated */],
    "quality_scores": [0.8, 0.2, 0.6, ...],
    "epochs": 10,
    "learning_rate": 0.001
}

// Receive
{
    "type": "train_complete",
    "n_samples": 100,
    "ensemble_val_loss": 0.0234,
    "is_trained": true
}
```

### Save Model

```javascript
// Send
{
    "type": "save",
    "path": "models/surrogate/checkpoint.pt"
}
```

### Get Status

```javascript
// Send
{ "type": "status" }

// Receive
{
    "type": "status",
    "is_trained": true,
    "n_training_samples": 1234,
    "input_dim": 64,
    "n_members": 5
}
```

## Genome Features (64D)

Features extracted from CPPN+DSP genomes:

### CPPN (waveNetwork) - 32 features
- **Topology** (8): node count, connection count, hidden nodes, depth, density
- **Activation distribution** (11): BipolarSigmoid, Sine, Sine2, sawtooth, etc.
- **Weight statistics** (8): mean, std, min, max, skewness, sparsity
- **Structure** (5): layer spread, hidden ratio, fan-out, modulation ratio

### DSP (asNEATPatch) - 32 features
- **Topology** (6): node count, connection count, evolution history
- **Node types** (8): GainNode, NoteOscillatorNode, OscillatorNode, etc.
- **Envelope params** (8): attack, decay, sustain, release statistics
- **Weights** (6): connection weight statistics
- **Oscillator types** (4): sine, square, sawtooth, triangle distribution

## Integration with QD Loop

The surrogate service fits into the QD evaluation pipeline:

1. **Generate candidates** (mutation/crossover)
2. **Extract features** → 64D vector
3. **Surrogate prediction** → quality estimate + uncertainty
4. **Selective evaluation**: Only render audio for high-predicted-quality or high-uncertainty candidates
5. **Online training**: Feed back ground truth quality to improve surrogate

### JavaScript Integration Module

The `surrogate-quality-prediction.js` module in `kromosynth-cli/cli-app/projection/` provides a complete integration layer:

```javascript
import { 
  initializeSurrogateService, 
  filterCandidatesWithSurrogate,
  collectTrainingData,
  maybeRetrainSurrogate,
  getSurrogateStatus,
  shutdownSurrogateService
} from './projection/surrogate-quality-prediction.js';
```

#### Key Functions

| Function | Purpose |
|----------|----------|
| `initializeSurrogateService(config)` | Connect to WebSocket service |
| `predictQuality(genomeString)` | Single genome quality prediction |
| `predictQualityBatch(genomeStrings)` | Batch prediction for efficiency |
| `filterCandidatesWithSurrogate(candidates)` | Pre-evaluation filtering |
| `collectTrainingData(genomeString, actualQuality)` | Gather training samples |
| `maybeRetrainSurrogate(generation, force)` | Periodic/forced retraining |
| `getSurrogateStatus()` | Get service status and accuracy metrics |

#### Configuration in Evolution Run Config

```jsonc
{
  "surrogateConfig": {
    "enabled": true,
    "serviceUrl": "ws://localhost:32070",
    "filterEnabled": true,
    "qualityThreshold": 0.3,      // Minimum predicted quality to proceed
    "uncertaintyThreshold": 0.15, // High uncertainty passes (exploration)
    "trainingBatchSize": 50,      // Samples before retraining
    "retrainingFrequency": 100    // Generations between retrains
  }
}
```

#### Integration in quality-diversity-search.js

The module is wired into the QD loop as follows:

1. **Initialization** (after server setup):
```javascript
const surrogateReady = await initializeSurrogateService(surrogateConfig);
```

2. **Training data collection** (end of each generation):
```javascript
// Collect elite scores for training
for (const [classKey, cell] of cellsWithElites) {
  const elite = cell.elts[0];
  const genomeString = await readGenomeAndMetaFromDisk(evolutionRunId, elite.g, evoRunDirPath);
  collectTrainingData(genomeString, elite.s, undefined);
}
```

3. **Periodic retraining** (every N generations):
```javascript
if (eliteMap.generationNumber % retrainingFrequency === 0) {
  await maybeRetrainSurrogate(eliteMap.generationNumber, true);
  const status = getSurrogateStatus();
  console.log(`MAE: ${status.rollingMAE}, Correlation: ${status.rollingCorrelation}`);
}
```

4. **Optional: Pre-evaluation filtering** (after mutation, before render):
```javascript
const { passed, filtered, predictions } = await filterCandidatesWithSurrogate(candidates);
// Only evaluate 'passed' candidates, skip 'filtered'
```

### Example Integration (Raw JavaScript)

```javascript
// After rendering and evaluating a batch
const trainRequest = {
    type: 'train',
    features_batch: evaluatedGenomes.map(g => extractFeatures(g)),
    quality_scores: evaluatedGenomes.map(g => g.evaluatedQuality),
    epochs: 5
};

// Send to surrogate service
ws.send(JSON.stringify(trainRequest));
```

## Uncertainty-Based Filtering

Use uncertainty to decide what to evaluate:

```javascript
const { quality, uncertainty } = await predict(genome);

// High confidence good → probably skip expensive evaluation
// High confidence bad → definitely skip
// High uncertainty → evaluate to improve surrogate

const shouldEvaluate = 
    uncertainty > UNCERTAINTY_THRESHOLD ||
    quality > QUALITY_THRESHOLD;
```

## Performance

- **Inference**: ~1ms single, ~15ms for 50-sample batch
- **Training**: ~5s for 100 samples × 10 epochs
- **Memory**: ~10MB for 5-member ensemble
