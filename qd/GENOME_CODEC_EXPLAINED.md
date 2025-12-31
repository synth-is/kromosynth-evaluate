# GenomeCodec: Bridging Kromosynth and pyribs

## The Problem

### Kromosynth Genome Format

Kromosynth uses **CPPN (Compositional Pattern Producing Networks)** to generate audio synthesis graphs. These genomes are complex, variable-structure representations:

```javascript
// Example kromosynth genome (simplified)
{
  "id": "abc123",
  "nodes": [
    {"id": "n1", "type": "sine", "activation": "linear"},
    {"id": "n2", "type": "gain", "activation": "tanh"},
    {"id": "n3", "type": "delay", "activation": "linear"}
  ],
  "connections": [
    {"from": "n1", "to": "n2", "weight": 0.8},
    {"from": "n2", "to": "n3", "weight": 0.3}
  ],
  "parameters": {
    "frequency": 440.0,
    "delayTime": 0.25
  }
}
```

**Key characteristics**:
- **Variable structure**: Different genomes can have different numbers of nodes/connections
- **Hierarchical**: Nested dictionaries and arrays
- **Graph-based**: Represents a network topology
- **Extensible**: Can grow or shrink during mutation

### pyribs Archive Requirements

pyribs (the QD library) requires solutions in a very different format:

```python
# What pyribs expects
solution = np.array([0.42, 0.78, 0.13, ...])  # Fixed-length vector
```

**Requirements**:
- **Fixed dimensionality**: Every solution must have the same length (e.g., always 100 dimensions)
- **Numeric arrays**: Pure numpy arrays, not dictionaries or graphs
- **Consistent shape**: Archive cells are fixed-size containers
- **CMA-ES compatible**: Must work with Covariance Matrix Adaptation

### The Fundamental Mismatch

```
Kromosynth                          pyribs
-----------                         --------
Variable length    ✗   ←→   ✓    Fixed length
Complex structures ✗   ←→   ✓    Flat arrays
Graphs            ✗   ←→   ✓    Vectors
Extensible        ✗   ←→   ✓    Rigid shape
```

**This creates a problem**: How do we use kromosynth's flexible genomes with pyribs' rigid archive?

## The Solution: GenomeCodec

The **GenomeCodec** is a translator that converts between these two representations:

```
encode()                    decode()
Genome  ──────────→  Solution  ──────────→  Genome
(complex)          (fixed array)           (complex)
```

### Strategy 1: ID-Based Reference (GenomeIDCodec)

**Approach**: Don't try to fit the genome into a vector. Instead, store it externally and use an ID as the solution.

```python
# Store genome in file system
genomes/
  genome_0.json     # Original kromosynth genome
  genome_1.json
  genome_2.json
  ...

# pyribs archive stores just IDs
solution_0 = [0.0]  # Points to genome_0.json
solution_1 = [1.0]  # Points to genome_1.json
solution_2 = [2.0]  # Points to genome_2.json
```

**How it works**:

1. **Encode** (genome → solution):
   ```python
   genome = {nodes: [...], connections: [...], ...}

   # Save genome to disk
   save_to_file("genome_42.json", genome)

   # Return ID as solution
   solution = np.array([42.0])  # Just the ID!
   ```

2. **Decode** (solution → genome):
   ```python
   solution = np.array([42.0])

   # Extract ID
   genome_id = int(solution[0])  # 42

   # Load genome from disk
   genome = load_from_file("genome_42.json")
   ```

**Advantages**:
- ✅ **Handles any genome complexity** - no size limits
- ✅ **Simple and robust** - straightforward mapping
- ✅ **Preserves all structure** - no information loss
- ✅ **Works with variable structures** - each genome can be different

**Trade-offs**:
- ⚠️ **External storage required** - genomes stored separately
- ⚠️ **CMA-ES limitation** - can't optimize genome parameters directly (but this is okay for kromosynth's mutation-based evolution)

### Strategy 2: Parameter Flattening (ParameterVectorCodec)

**Approach**: If genome structure is fixed, flatten all numeric parameters into a vector.

```python
# Fixed structure genome
genome = {
  "nodes": 3,           # Always 3 nodes
  "connections": 2,     # Always 2 connections
  "params": {
    "freq": 440.0,
    "gain": 0.8,
    "delay": 0.25
  }
}

# Flatten to vector
solution = np.array([440.0, 0.8, 0.25])  # Just the parameters
```

**How it works**:

1. **Encode**:
   ```python
   genome = {"params": {"freq": 440.0, "gain": 0.8, "delay": 0.25}}

   # Extract parameters in fixed order
   solution = np.array([
       genome["params"]["freq"],
       genome["params"]["gain"],
       genome["params"]["delay"]
   ])  # [440.0, 0.8, 0.25]
   ```

2. **Decode**:
   ```python
   solution = np.array([440.0, 0.8, 0.25])

   # Reconstruct genome with fixed structure
   genome = {
       "params": {
           "freq": solution[0],
           "gain": solution[1],
           "delay": solution[2]
       }
   }
   ```

**Advantages**:
- ✅ **CMA-ES can optimize** - parameters directly evolved
- ✅ **Compact** - no external storage needed
- ✅ **Fast** - direct parameter manipulation

**Trade-offs**:
- ⚠️ **Requires fixed structure** - genome topology can't change
- ⚠️ **Less flexible** - can't add/remove nodes during evolution
- ⚠️ **Implementation complexity** - need to know exact genome structure

## Which Strategy to Use?

### For kromosynth: Use GenomeIDCodec (Strategy 1)

**Reason**: Kromosynth's genomes are variable-structure by design:
- Genomes can have different numbers of nodes
- Connections can be added/removed through mutation
- Graph topology is part of the evolutionary search

**Implementation**:
```python
from qd.genome_codec import GenomeIDCodec

codec = GenomeIDCodec(
    solution_dim=1,  # Just need space for ID
    genome_dir="./genomes"
)

# During evolution
solution = codec.encode(kromosynth_genome)  # Returns [genome_id]
archive.add(solution, fitness, behavior_descriptor)

# Later retrieval
elite = archive.sample_elites(1)[0]
genome = codec.decode(elite['solution'])  # Reconstructs full genome
```

### For fixed-parameter evolution: Use ParameterVectorCodec (Strategy 2)

If you ever want to:
- Fix the genome structure
- Let CMA-ES optimize just parameters
- Avoid external genome storage

Then ParameterVectorCodec becomes useful.

## Integration with pyribs Workflow

Here's how GenomeCodec fits into the full QD loop:

```python
# Setup
archive = ArchiveManager(solution_dim=1, bd_dim=6, num_cells=10000)
emitters = EmitterManager(archive, num_emitters=5)
codec = GenomeIDCodec(genome_dir="./genomes")

# QD Loop
for generation in range(1000):
    # 1. ASK: Get solution vectors from emitters
    solutions, emitter_ids = emitters.ask()
    # solutions shape: (180, 1)  <- 180 IDs

    # 2. DECODE: Convert IDs back to kromosynth genomes
    genomes = [codec.decode(sol) for sol in solutions]

    # 3. EVALUATE: Render audio and extract features
    audio_buffers = [render_genome(g) for g in genomes]
    clap_embeddings = extract_clap(audio_buffers)
    behavior_descriptors = project_to_bd(clap_embeddings)  # Phase 3
    fitnesses = evaluate_quality(audio_buffers)

    # 4. TELL: Report results back to archive
    emitters.tell(solutions, fitnesses, behavior_descriptors)

    # Note: solutions (IDs) go in/out of archive,
    # but genomes stay in files
```

## Why This Matters

Without GenomeCodec, we'd have to choose between:

1. **Abandoning pyribs** - Rewrite all QD infrastructure for kromosynth's format
2. **Abandoning flexibility** - Force kromosynth genomes into fixed-length vectors (losing variable structure)
3. **Manual translation** - Write brittle, one-off conversion code

GenomeCodec provides a **clean abstraction** that:
- ✅ Keeps kromosynth genomes flexible
- ✅ Lets pyribs use efficient CVT-MAP-Elites
- ✅ Maintains separation of concerns
- ✅ Allows different encoding strategies as needed

## Technical Details

### GenomeIDCodec Implementation

**File structure**:
```
genomes/
  genome_0.json
  genome_1.json
  genome_2.json
  ...
```

**Storage format**:
```json
{
  "nodes": [...],
  "connections": [...],
  "parameters": {...},
  "_codec_id": 42
}
```

**ID assignment**:
- Auto-incrementing counter
- Persisted across codec instances
- Recovers from existing files on restart

### Memory & Performance

**GenomeIDCodec**:
- Archive stores: ~8 bytes per solution (just float64 ID)
- Disk usage: ~1-10 KB per genome JSON file
- For 10,000 elites: ~10-100 MB total

**Alternatives considered**:
- **Pickle serialization**: Less human-readable, versioning issues
- **Database storage**: Overkill for current scale
- **Hash-based lookup**: Collision risk, no sequential access

## Future Extensions

### Hybrid Approach

Combine both strategies:
- Use **ID codec** for complex genome structure
- Use **parameter codec** for specific parameter tuning

```python
class HybridCodec:
    def encode(self, genome):
        structure_id = save_structure(genome)
        params = extract_parameters(genome)
        return np.concatenate([[structure_id], params])
```

### Compressed Encoding

For very large archives:
- Genome deduplication
- Compression of similar structures
- Delta encoding (store differences from parent)

### Distributed Storage

For cluster environments:
- Shared genome database
- Network file system
- Cloud storage integration

## Summary

**GenomeCodec exists because**:
- Kromosynth genomes are variable-structure graphs
- pyribs requires fixed-length numeric vectors
- We need a bridge between these representations

**The ID-based approach**:
- Stores genomes externally as JSON
- Uses genome ID as the solution vector
- Preserves all genome flexibility
- Recommended for kromosynth

**This enables**:
- Using pyribs' efficient QD algorithms
- Keeping kromosynth's flexible evolution
- Clean separation of concerns
- Extensible architecture for future needs
