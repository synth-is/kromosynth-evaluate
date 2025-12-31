"""
Genome Feature Extraction for Surrogate Model Input.

Extracts fixed-length feature vectors from variable-structure CPPN+DSP genomes.
Features are normalized to [0, 1] range for neural network input.

## Purpose

The surrogate model needs to predict quality BEFORE rendering audio.
This requires extracting informative features from genome structure alone.

## Genome Structure

CPPN+DSP genomes have two main components:

1. **waveNetwork** (CPPN - Compositional Pattern Producing Network):
   - Variable number of nodes (Bias, Input, Output, Hidden)
   - Variable number of connections with weights
   - Activation functions: BipolarSigmoid, Sine, Sine2, sawtooth, NullFn, etc.

2. **asNEATPatch** (DSP graph - Audio Signal Processing):
   - Audio nodes: GainNode, NoteOscillatorNode, OscillatorNode, OutNode
   - Connections with weights and parameter modulation
   - Envelope parameters (attack, decay, sustain, release)

## Feature Categories

- **Topology**: Node/connection counts, depth, density
- **Parameters**: Weight statistics, envelope aggregates
- **Structure**: Activation function distribution, node type ratios
- **Complexity**: Connectivity measures, modularity indicators

Total output dimension: 64 features (configurable)
"""

import numpy as np
import json
from typing import Dict, Any, Union, List, Optional, Tuple
from collections import Counter
import warnings


# Known activation functions (extend as needed)
ACTIVATION_FUNCTIONS = [
    'NullFn', 'BipolarSigmoid', 'Sine', 'Sine2', 'sawtooth',
    'Gaussian', 'Linear', 'Step', 'Tanh', 'ReLU', 'Cos'
]

# Known node types
NODE_TYPES = ['Bias', 'Input', 'Output', 'Hidden']

# Known DSP node types
DSP_NODE_TYPES = [
    'GainNode', 'NoteOscillatorNode', 'OscillatorNode', 'OutNode',
    'BiquadFilterNode', 'DelayNode', 'WaveShaperNode', 'ConvolverNode'
]

# Oscillator types
OSCILLATOR_TYPES = ['sine', 'square', 'sawtooth', 'triangle', 'custom']


class GenomeFeatureExtractor:
    """
    Extracts fixed-length, normalized feature vectors from CPPN+DSP genomes.
    
    Features are designed to be informative for quality prediction:
    - Network complexity correlates with sound richness
    - Parameter distributions affect timbral characteristics
    - Structural patterns influence musical suitability
    """
    
    # Feature dimension breakdown
    CPPN_TOPOLOGY_DIM = 8      # Node/connection counts, depth, density
    CPPN_ACTIVATION_DIM = 11  # Activation function distribution
    CPPN_WEIGHTS_DIM = 8       # Weight statistics
    CPPN_STRUCTURE_DIM = 5     # Layer distribution, modularity
    
    DSP_TOPOLOGY_DIM = 6       # Node/connection counts
    DSP_NODE_TYPES_DIM = 8     # Node type distribution
    DSP_ENVELOPE_DIM = 8       # Envelope parameter statistics
    DSP_WEIGHTS_DIM = 6        # Connection weight statistics
    DSP_OSC_DIM = 4            # Oscillator type distribution
    
    TOTAL_DIM = (CPPN_TOPOLOGY_DIM + CPPN_ACTIVATION_DIM + CPPN_WEIGHTS_DIM + 
                 CPPN_STRUCTURE_DIM + DSP_TOPOLOGY_DIM + DSP_NODE_TYPES_DIM + 
                 DSP_ENVELOPE_DIM + DSP_WEIGHTS_DIM + DSP_OSC_DIM)  # = 64
    
    def __init__(
        self,
        normalize: bool = True,
        feature_dim: Optional[int] = None
    ):
        """
        Initialize feature extractor.
        
        Args:
            normalize: Whether to normalize features to [0, 1]
            feature_dim: Override feature dimension (for compatibility)
        """
        self.normalize = normalize
        self.feature_dim = feature_dim or self.TOTAL_DIM
        
        # Running statistics for adaptive normalization
        self._stats = {
            'weight_max': 5.0,      # Connection weights can be large
            'freq_max': 20000.0,    # Max frequency for oscillators
            'max_nodes': 100,       # Expected max nodes
            'max_connections': 500, # Expected max connections
            'max_depth': 20,        # Expected max layer depth
        }
    
    def extract(self, genome: Union[Dict, str]) -> np.ndarray:
        """
        Extract feature vector from genome.
        
        Args:
            genome: CPPN+DSP genome (dict or JSON string)
            
        Returns:
            Feature vector of shape (feature_dim,)
        """
        if isinstance(genome, str):
            try:
                genome = json.loads(genome)
            except json.JSONDecodeError:
                warnings.warn("Failed to parse genome JSON, returning zeros")
                return np.zeros(self.feature_dim)
        
        # Handle nested genome structure
        if 'genome' in genome:
            genome_data = genome['genome']
        else:
            genome_data = genome
        
        # Extract features from each component
        cppn_features = self._extract_cppn_features(genome_data.get('waveNetwork', {}))
        dsp_features = self._extract_dsp_features(genome_data.get('asNEATPatch', '{}'))
        
        # Concatenate all features
        features = np.concatenate([cppn_features, dsp_features])
        
        # Ensure correct dimension
        if len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        elif len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        
        return features
    
    def _extract_cppn_features(self, wave_network: Dict) -> np.ndarray:
        """Extract features from CPPN (waveNetwork)."""
        features = []
        
        # Handle nested 'offspring' structure
        if 'offspring' in wave_network:
            network = wave_network['offspring']
        else:
            network = wave_network
        
        nodes = network.get('nodes', [])
        connections = network.get('connections', [])
        
        # === Topology Features (8) ===
        n_nodes = len(nodes)
        n_connections = len(connections)
        n_hidden = sum(1 for n in nodes if n.get('nodeType') == 'Hidden')
        n_input = sum(1 for n in nodes if n.get('nodeType') == 'Input')
        n_output = sum(1 for n in nodes if n.get('nodeType') == 'Output')
        n_bias = sum(1 for n in nodes if n.get('nodeType') == 'Bias')
        
        # Network depth (max layer)
        layers = [n.get('layer', 0) for n in nodes]
        max_depth = max(layers) if layers else 0
        
        # Connectivity density
        max_possible = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
        density = n_connections / max_possible if max_possible > 0 else 0
        
        topology = [
            self._normalize_count(n_nodes, self._stats['max_nodes']),
            self._normalize_count(n_connections, self._stats['max_connections']),
            self._normalize_count(n_hidden, self._stats['max_nodes']),
            self._normalize_count(n_input, 10),   # Usually small
            self._normalize_count(n_output, 30),  # Up to ~20 outputs seen
            self._normalize_count(n_bias, 5),     # Usually 1
            self._normalize_count(max_depth, self._stats['max_depth']),
            min(density, 1.0),  # Already normalized
        ]
        features.extend(topology)
        
        # === Activation Function Distribution (11) ===
        activation_counts = Counter(n.get('activationFunction', 'NullFn') for n in nodes)
        activation_dist = [
            self._normalize_count(activation_counts.get(af, 0), n_nodes + 1)
            for af in ACTIVATION_FUNCTIONS
        ]
        features.extend(activation_dist)
        
        # === Weight Statistics (8) ===
        weights = [c.get('weight', 0.0) for c in connections]
        if weights:
            w_array = np.array(weights)
            weight_stats = [
                self._normalize_weight(np.mean(w_array)),
                self._normalize_weight(np.std(w_array)),
                self._normalize_weight(np.min(w_array)),
                self._normalize_weight(np.max(w_array)),
                self._normalize_weight(np.median(w_array)),
                # Skewness indicator
                np.tanh((np.mean(w_array > 0) - 0.5) * 2),
                # Sparsity (weights near zero)
                np.mean(np.abs(w_array) < 0.1),
                # Large weight ratio
                np.mean(np.abs(w_array) > 1.0),
            ]
        else:
            weight_stats = [0.5, 0.0, 0.5, 0.5, 0.5, 0.0, 1.0, 0.0]
        features.extend(weight_stats)
        
        # === Structural Features (5) ===
        # Layer distribution (how spread out are nodes across layers?)
        unique_layers = len(set(layers)) if layers else 1
        layer_spread = unique_layers / (max_depth + 1) if max_depth > 0 else 1.0
        
        # Hidden node ratio
        hidden_ratio = n_hidden / n_nodes if n_nodes > 0 else 0.0
        
        # Average fan-out (connections per source node)
        source_counts = Counter(c.get('sourceID') for c in connections)
        avg_fanout = np.mean(list(source_counts.values())) if source_counts else 0
        
        # Modulation connection ratio
        mod_connections = sum(1 for c in connections if c.get('modConnection', 0) > 0)
        mod_ratio = mod_connections / (n_connections + 1)
        
        # Mutated connection ratio
        mutated = sum(1 for c in connections if c.get('isMutated', False))
        mutated_ratio = mutated / (n_connections + 1)
        
        structure = [
            min(layer_spread, 1.0),
            min(hidden_ratio, 1.0),
            self._normalize_count(avg_fanout, 20),
            min(mod_ratio, 1.0),
            min(mutated_ratio, 1.0),
        ]
        features.extend(structure)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_dsp_features(self, as_neat_patch: Union[str, Dict]) -> np.ndarray:
        """Extract features from DSP graph (asNEATPatch)."""
        features = []
        
        # Parse if string
        if isinstance(as_neat_patch, str):
            try:
                patch = json.loads(as_neat_patch)
            except json.JSONDecodeError:
                # Return zeros if parsing fails
                return np.zeros(self.DSP_TOPOLOGY_DIM + self.DSP_NODE_TYPES_DIM + 
                               self.DSP_ENVELOPE_DIM + self.DSP_WEIGHTS_DIM + 
                               self.DSP_OSC_DIM)
        else:
            patch = as_neat_patch
        
        # Parse nodes (stored as JSON strings in array)
        nodes = []
        for node_str in patch.get('nodes', []):
            if isinstance(node_str, str):
                try:
                    nodes.append(json.loads(node_str))
                except json.JSONDecodeError:
                    continue
            else:
                nodes.append(node_str)
        
        # Parse connections
        connections = []
        for conn_str in patch.get('connections', []):
            if isinstance(conn_str, str):
                try:
                    connections.append(json.loads(conn_str))
                except json.JSONDecodeError:
                    continue
            else:
                connections.append(conn_str)
        
        n_nodes = len(nodes)
        n_connections = len(connections)
        
        # === DSP Topology Features (6) ===
        topology = [
            self._normalize_count(n_nodes, 20),
            self._normalize_count(n_connections, 50),
            self._normalize_count(len(patch.get('evolutionHistory', [])), 20),
            # Generation indicator
            self._normalize_count(patch.get('generation', 0), 100),
            # Enabled connection ratio
            np.mean([c.get('enabled', True) for c in connections]) if connections else 1.0,
            # Has parameter modulation
            float(any(c.get('targetParameter') for c in connections)),
        ]
        features.extend(topology)
        
        # === Node Type Distribution (8) ===
        node_names = [n.get('name', '') for n in nodes]
        node_type_dist = [
            self._normalize_count(node_names.count(nt), n_nodes + 1)
            for nt in DSP_NODE_TYPES
        ]
        features.extend(node_type_dist)
        
        # === Envelope Parameters (8) ===
        # Collect envelope params from oscillator nodes
        attack_durs = []
        decay_durs = []
        sustain_durs = []
        release_durs = []
        attack_vols = []
        sustain_vols = []
        
        for node in nodes:
            if 'attackDuration' in node:
                attack_durs.append(node.get('attackDuration', 0))
                decay_durs.append(node.get('decayDuration', 0))
                sustain_durs.append(node.get('sustainDuration', 0))
                release_durs.append(node.get('releaseDuration', 0))
                attack_vols.append(node.get('attackVolume', 1))
                sustain_vols.append(node.get('sustainVolume', 1))
        
        envelope_stats = [
            np.mean(attack_durs) if attack_durs else 0.5,
            np.mean(decay_durs) if decay_durs else 0.5,
            np.mean(sustain_durs) if sustain_durs else 0.5,
            np.mean(release_durs) if release_durs else 0.5,
            np.mean(attack_vols) if attack_vols else 1.0,
            np.mean(sustain_vols) if sustain_vols else 1.0,
            np.std(attack_durs) if len(attack_durs) > 1 else 0.0,
            np.std(release_durs) if len(release_durs) > 1 else 0.0,
        ]
        # Clamp envelope stats to [0, 1]
        envelope_stats = [min(max(v, 0), 1) for v in envelope_stats]
        features.extend(envelope_stats)
        
        # === Connection Weight Statistics (6) ===
        weights = [c.get('weight', 0.0) for c in connections]
        if weights:
            w_array = np.array(weights)
            # For DSP weights, some can be large (frequency modulation)
            weight_stats = [
                np.tanh(np.mean(w_array) / 100),   # Normalize large values
                np.tanh(np.std(w_array) / 100),
                np.tanh(np.min(w_array) / 100),
                np.tanh(np.max(w_array) / 100),
                np.mean(np.abs(w_array) < 1.0),    # Small weight ratio
                np.mean(np.abs(w_array) > 100),   # Large weight ratio (FM)
            ]
            weight_stats = [(v + 1) / 2 for v in weight_stats[:4]] + weight_stats[4:]
        else:
            weight_stats = [0.5, 0.0, 0.5, 0.5, 1.0, 0.0]
        features.extend(weight_stats)
        
        # === Oscillator Type Distribution (4) ===
        osc_types = []
        for node in nodes:
            if 'type' in node:
                osc_types.append(node['type'])
        
        osc_dist = [
            osc_types.count('sine') / (len(osc_types) + 1),
            osc_types.count('square') / (len(osc_types) + 1),
            osc_types.count('sawtooth') / (len(osc_types) + 1),
            osc_types.count('triangle') / (len(osc_types) + 1),
        ]
        features.extend(osc_dist)
        
        return np.array(features, dtype=np.float32)
    
    def _normalize_count(self, value: float, max_val: float) -> float:
        """Normalize count to [0, 1] using soft saturation."""
        if max_val <= 0:
            return 0.0
        return min(value / max_val, 1.0)
    
    def _normalize_weight(self, value: float) -> float:
        """Normalize weight using tanh then shift to [0, 1]."""
        # tanh maps to [-1, 1], shift to [0, 1]
        return (np.tanh(value / self._stats['weight_max']) + 1) / 2
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names for interpretability."""
        names = []
        
        # CPPN Topology
        names.extend([
            'cppn_n_nodes', 'cppn_n_connections', 'cppn_n_hidden',
            'cppn_n_input', 'cppn_n_output', 'cppn_n_bias',
            'cppn_max_depth', 'cppn_density'
        ])
        
        # CPPN Activation
        names.extend([f'cppn_act_{af}' for af in ACTIVATION_FUNCTIONS])
        
        # CPPN Weights
        names.extend([
            'cppn_w_mean', 'cppn_w_std', 'cppn_w_min', 'cppn_w_max',
            'cppn_w_median', 'cppn_w_skew', 'cppn_w_sparse', 'cppn_w_large'
        ])
        
        # CPPN Structure
        names.extend([
            'cppn_layer_spread', 'cppn_hidden_ratio', 'cppn_avg_fanout',
            'cppn_mod_ratio', 'cppn_mutated_ratio'
        ])
        
        # DSP Topology
        names.extend([
            'dsp_n_nodes', 'dsp_n_connections', 'dsp_evolution_len',
            'dsp_generation', 'dsp_enabled_ratio', 'dsp_has_param_mod'
        ])
        
        # DSP Node Types
        names.extend([f'dsp_type_{nt}' for nt in DSP_NODE_TYPES])
        
        # DSP Envelope
        names.extend([
            'dsp_env_attack', 'dsp_env_decay', 'dsp_env_sustain', 'dsp_env_release',
            'dsp_env_attack_vol', 'dsp_env_sustain_vol',
            'dsp_env_attack_std', 'dsp_env_release_std'
        ])
        
        # DSP Weights
        names.extend([
            'dsp_w_mean', 'dsp_w_std', 'dsp_w_min', 'dsp_w_max',
            'dsp_w_small_ratio', 'dsp_w_large_ratio'
        ])
        
        # DSP Oscillator Types
        names.extend(['dsp_osc_sine', 'dsp_osc_square', 'dsp_osc_sawtooth', 'dsp_osc_triangle'])
        
        return names


def extract_features_batch(
    genomes: List[Union[Dict, str]],
    extractor: Optional[GenomeFeatureExtractor] = None
) -> np.ndarray:
    """
    Extract features from a batch of genomes.
    
    Args:
        genomes: List of genome dicts or JSON strings
        extractor: Optional pre-configured extractor
        
    Returns:
        Feature matrix of shape (n_genomes, feature_dim)
    """
    if extractor is None:
        extractor = GenomeFeatureExtractor()
    
    features = [extractor.extract(g) for g in genomes]
    return np.stack(features)


# Convenience function for codec integration
def genome_to_features(genome: Union[Dict, str]) -> np.ndarray:
    """
    Simple function to convert genome to features.
    Uses default extractor settings.
    """
    extractor = GenomeFeatureExtractor()
    return extractor.extract(genome)


if __name__ == "__main__":
    # Test with example genome structure
    example_genome = {
        "genome": {
            "waveNetwork": {
                "offspring": {
                    "nodes": [
                        {"gid": "0", "activationFunction": "NullFn", "nodeType": "Bias", "layer": 0},
                        {"gid": "1", "activationFunction": "NullFn", "nodeType": "Input", "layer": 0},
                        {"gid": "2", "activationFunction": "NullFn", "nodeType": "Input", "layer": 0},
                        {"gid": "3", "activationFunction": "BipolarSigmoid", "nodeType": "Output", "layer": 10},
                        {"gid": "4", "activationFunction": "Sine", "nodeType": "Hidden", "layer": 5},
                    ],
                    "connections": [
                        {"gid": "c1", "weight": 0.5, "sourceID": "1", "targetID": "4"},
                        {"gid": "c2", "weight": -0.3, "sourceID": "4", "targetID": "3"},
                    ]
                }
            },
            "asNEATPatch": json.dumps({
                "nodes": [
                    json.dumps({"name": "NoteOscillatorNode", "type": "sine", "attackDuration": 0.5}),
                    json.dumps({"name": "OutNode", "id": 0}),
                ],
                "connections": [
                    json.dumps({"sourceNode": "osc1", "targetNode": 0, "weight": 0.8}),
                ]
            })
        }
    }
    
    extractor = GenomeFeatureExtractor()
    features = extractor.extract(example_genome)
    
    print(f"Feature dimension: {len(features)}")
    print(f"Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"\nFeature names ({len(extractor.get_feature_names())}):")
    for name, val in zip(extractor.get_feature_names(), features):
        if val > 0:
            print(f"  {name}: {val:.3f}")
