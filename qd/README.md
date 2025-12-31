# Quality Diversity (QD) Module

CVT-MAP-Elites with CMA-MAE emitters for kromosynth evolutionary sound discovery.

## Overview

This module provides pyribs-based Quality Diversity search for kromosynth, enabling efficient exploration of the sound behavior space using:

- **CVT-MAP-Elites**: Archive of elite solutions organized by behavior descriptors
- **CMA-MAE**: Multiple CMA-ES emitters for gradient-free optimization
- **GenomeCodec**: Bridge between kromosynth's variable genomes and pyribs' fixed vectors

## Components

- **`archive_manager.py`**: CVT archive wrapper with persistence and statistics
- **`emitter_manager.py`**: CMA-MAE emitter coordination for ask/tell pattern
- **`genome_codec.py`**: Genome encoding/decoding for pyribs compatibility
- **`ws_pyribs_service.py`**: WebSocket/REST service (pending)

## Quick Start

### Python API

```python
from qd import ArchiveManager, EmitterManager
from qd.genome_codec import GenomeIDCodec

# Setup
archive = ArchiveManager(
    solution_dim=1,      # Just need space for genome ID
    bd_dim=6,            # 6D behavior descriptors (from QDHF projection)
    num_cells=10000,     # 10K niches in behavior space
    ranges=[(0, 1)] * 6  # BD range [0,1] per dimension
)

emitters = EmitterManager(
    archive=archive,
    num_emitters=5,      # 5 parallel CMA-ES instances
    sigma0=0.5,          # Initial step size
    batch_size=36        # 36 solutions per emitter = 180 total
)

codec = GenomeIDCodec(genome_dir="./genomes")

# QD Loop
for generation in range(1000):
    # Get candidate solutions
    solutions, emitter_ids = emitters.ask()  # (180, 1) array of IDs

    # Decode to kromosynth genomes
    genomes = [codec.decode(sol) for sol in solutions]

    # Evaluate (your rendering + feature extraction pipeline)
    fitnesses = evaluate_quality(genomes)
    bds = extract_behavior_descriptors(genomes)

    # Report results
    stats = emitters.tell(solutions, fitnesses, bds)

    if generation % 10 == 0:
        print(f"Gen {generation}: QD={stats['qd_score']:.1f}, "
              f"Coverage={stats['coverage']:.2%}")

# Save archive
archive.save("archive_final.pkl")
```

## GenomeCodec: The Key Abstraction

### Why It Exists

**The Problem**:
- Kromosynth genomes are variable-structure CPPN graphs (different numbers of nodes/connections)
- pyribs requires fixed-length numeric vectors for all solutions

**The Solution**:
GenomeCodec translates between these representations:

```
Kromosynth Genome          Solution Vector          pyribs Archive
(complex graph)      →     (fixed array)      →     (CVT cells)
    ↓                           ↓                         ↓
{nodes: [...],            [genome_id]              Cell 4237
 connections: [...],   →      ↓                    stores: [42.0]
 params: {...}}           genome_42.json
```

### ID-Based Encoding (Recommended)

```python
codec = GenomeIDCodec(solution_dim=1, genome_dir="./genomes")

# Encoding: genome → vector
genome = {nodes: [...], connections: [...], parameters: {...}}
solution = codec.encode(genome)  # Returns: [42.0] (genome saved to genomes/genome_42.json)

# Decoding: vector → genome
genome = codec.decode(solution)  # Loads from genomes/genome_42.json
```

**See**: `GENOME_CODEC_EXPLAINED.md` for detailed explanation

## Archive Statistics

```python
stats = archive.get_stats()
# {
#     "qd_score": 12453.7,      # Sum of all elite fitnesses
#     "coverage": 0.73,          # Fraction of cells filled
#     "max_fitness": 0.95,       # Best solution
#     "mean_fitness": 0.68,      # Average fitness
#     "num_elites": 7321,        # Number of solutions
#     "cells_filled": 7321,
#     "cells_total": 10000
# }
```

## Configuration

### Archive Parameters

```python
ArchiveManager(
    solution_dim=1,          # Dimensionality of solution vectors
    bd_dim=6,                # Behavior descriptor dimensions
    num_cells=10000,         # Number of CVT cells/niches
    ranges=[(0,1)] * 6,      # BD range per dimension
    seed=42                  # Random seed for reproducibility
)
```

### Emitter Parameters

```python
EmitterManager(
    archive=archive,
    num_emitters=5,          # Parallel CMA-ES instances
    sigma0=0.5,              # Initial step size
    batch_size=36,           # Solutions per emitter per ask()
    ranker="imp",            # "imp" (improvement) or "obj" (objective)
    selection_rule="filter", # Parent selection strategy
    restart_rule="basic"     # Restart strategy
)
```

### Typical Configurations

**Fast exploration** (prototype):
- num_cells: 1,000
- num_emitters: 3
- batch_size: 20
- Total: 60 solutions/generation

**Standard** (recommended):
- num_cells: 10,000
- num_emitters: 5
- batch_size: 36
- Total: 180 solutions/generation

**Intensive** (production):
- num_cells: 50,000
- num_emitters: 10
- batch_size: 50
- Total: 500 solutions/generation

## Archive Persistence

```python
# Save
archive.save("archive_gen1000.pkl")

# Load
archive.load("archive_gen1000.pkl")
```

## Remapping (for Updated Projections)

When the QDHF projection model is retrained:

```python
# Get all current elites
data = archive.archive.data(return_type="dict")
genomes = [codec.decode(sol) for sol in data["solution"]]

# Re-extract features with new projection model
new_bds = extract_behavior_descriptors_v2(genomes)

# Remap archive
archive.remap(new_bds, new_ranges=[(0, 1)] * 6)

# Reset emitters with new archive state
emitters.reset_emitters()
```

## Integration with kromosynth-cli

The full pipeline (Phase 4):

```
kromosynth-cli              kromosynth-evaluate
--------------              --------------------

QD Loop:                    Services:
  ask() ──────────────────→ pyribs service (port 32052)
      ←──────────────────── [genome_ids]

  decode IDs to genomes
  render to audio
  extract CLAP ──────────→  CLAP service (port 32051)
      ←──────────────────── [512D embeddings]

  project to BDs ─────────→ Projection service
      ←──────────────────── [6D behavior descriptors]

  evaluate quality

  tell() ─────────────────→ pyribs service
      (solutions, fitness, BDs)
```

## Performance

### Archive Operations

- Add single: ~0.1ms
- Add batch (100): ~5ms
- Get stats: ~1ms
- Save/load: ~100-500ms (depends on archive size)

### Memory Usage

- Archive (10K cells): ~80MB
- Genome storage (10K elites): ~10-100MB (depends on genome size)
- Total: ~100-200MB for typical use

### Expected QD Metrics

After 1,000 generations (180K evaluations):
- Coverage: 60-80% (6,000-8,000 cells filled)
- QD Score: Varies by domain (higher is better)
- Max Fitness: Approaches theoretical maximum

## Troubleshooting

### "Solution dimension mismatch"

Make sure all solutions have the same length:
```python
codec = GenomeIDCodec(solution_dim=1)  # All solutions will be [id]
```

### "No elites in archive"

Archive starts empty. Add initial solutions:
```python
# Generate random initial genomes
for i in range(100):
    genome = generate_random_genome()
    solution = codec.encode(genome)
    fitness = evaluate(genome)
    bd = extract_bd(genome)
    archive.add(solution, fitness, bd)
```

### Emitters not improving

- Check `sigma0` (try 0.1 to 1.0)
- Verify fitnesses are in [0, 1] range
- Ensure BDs are in specified ranges
- Try different ranker ("imp" vs "obj")

## References

- **pyribs**: https://docs.pyribs.org/
- **CMA-MAE paper**: https://arxiv.org/abs/2205.10752
- **CVT-MAP-Elites**: https://arxiv.org/abs/1610.05729
- **QDHF paper** (Phase 3): https://arxiv.org/abs/2310.12103

## File Structure

```
qd/
├── __init__.py                    # Module exports
├── archive_manager.py             # CVT archive wrapper
├── emitter_manager.py             # CMA-MAE emitters
├── genome_codec.py                # Genome encoding/decoding
├── ws_pyribs_service.py           # WebSocket service (pending)
├── README.md                      # This file
├── GENOME_CODEC_EXPLAINED.md      # Detailed codec explanation
└── PHASE2_PROGRESS.md             # Implementation status
```

## Next Steps

1. Implement WebSocket service (`ws_pyribs_service.py`)
2. Create test suite
3. Integrate with kromosynth-cli (Phase 4)
4. Connect with CLAP features (Phase 1) and QDHF projection (Phase 3)

## License

Same as kromosynth-evaluate
