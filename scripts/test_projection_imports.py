#!/usr/bin/env python3
"""
Test that QDHF projection modules import correctly.

Quick smoke test to verify all components are importable.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Testing QDHF projection imports...")
print()

# Test 1: Import projection network
print("1. Testing ProjectionNetwork import...")
try:
    from projection.qdhf.projection_network import (
        ProjectionNetwork,
        create_projection_network,
        small_projection_network,
        standard_projection_network,
        large_projection_network,
        deep_projection_network
    )
    print("   ✓ ProjectionNetwork imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 2: Import triplet generator
print("2. Testing ProxyTripletGenerator import...")
try:
    from projection.qdhf.proxy_triplet_generator import ProxyTripletGenerator
    print("   ✓ ProxyTripletGenerator imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Import trainer
print("3. Testing TripletTrainer import...")
try:
    from projection.qdhf.triplet_trainer import TripletTrainer
    print("   ✓ TripletTrainer imported")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 4: Import from package
print("4. Testing package-level imports...")
try:
    from projection.qdhf import (
        ProjectionNetwork,
        ProxyTripletGenerator,
        TripletTrainer
    )
    print("   ✓ Package imports work")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Create instances
print("5. Testing instance creation...")
try:
    import numpy as np

    # Create network
    network = standard_projection_network()
    print(f"   ✓ Network created ({network.get_num_parameters():,} parameters)")

    # Create triplet generator
    embeddings = np.random.randn(100, 512)
    generator = ProxyTripletGenerator(embeddings)
    print(f"   ✓ Generator created ({len(embeddings)} embeddings)")

    # Create trainer
    trainer = TripletTrainer(
        model=network,
        triplet_generator=generator
    )
    print(f"   ✓ Trainer created")

except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 6: Basic forward pass
print("6. Testing forward pass...")
try:
    import torch

    # Create input
    batch_size = 10
    x = torch.randn(batch_size, 512)

    # Forward pass
    network.eval()
    with torch.no_grad():
        output = network(x)

    # Validate output
    assert output.shape == (batch_size, 6), f"Wrong output shape: {output.shape}"
    assert torch.all((output >= 0) & (output <= 1)), "Output not in [0, 1] range"

    print(f"   ✓ Forward pass successful")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.3f}, {output.max():.3f}]")

except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 7: Triplet generation
print("7. Testing triplet generation...")
try:
    triplet_indices = generator.generate_batch(10)
    anchors, positives, negatives = generator.get_embeddings_for_triplets(triplet_indices)

    assert anchors.shape == (10, 512), f"Wrong anchor shape: {anchors.shape}"
    assert positives.shape == (10, 512), f"Wrong positive shape: {positives.shape}"
    assert negatives.shape == (10, 512), f"Wrong negative shape: {negatives.shape}"

    print(f"   ✓ Triplet generation successful")
    print(f"   Generated {len(triplet_indices)} triplets")

except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print()
print("=" * 60)
print("All import tests passed!")
print("=" * 60)
print()
print("Next steps:")
print("  1. Train a projection model:")
print("     python scripts/train_projection.py \\")
print("       --embeddings /path/to/embeddings.npy \\")
print("       --output models/projection/projection_v1.pt")
print()
print("  2. Start projection service:")
print("     ./projection/qdhf/start_projection_service.sh \\")
print("       --model models/projection/projection_v1.pt")
print()
print("  3. Test service:")
print("     python scripts/test_projection_service.py")
