#!/usr/bin/env python3
"""
Quick smoke test for CLAP extractor after version upgrade.
Tests basic functionality without extensive benchmarking.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from features.clap.clap_extractor import CLAPExtractor


def generate_sine_wave(frequency=440, duration=0.5, sample_rate=16000):
    """Generate a simple sine wave for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


def quick_test():
    """Quick smoke test."""
    print("=" * 60)
    print("CLAP v1.1.7 Quick Smoke Test")
    print("=" * 60)

    print("\n[1/3] Initializing extractor (CPU)...")
    start = time.time()
    extractor = CLAPExtractor(device='cpu')
    elapsed = time.time() - start
    print(f"✓ Initialized in {elapsed:.2f}s")

    print("\n[2/3] Extracting single embedding...")
    audio = generate_sine_wave(frequency=440, duration=0.5, sample_rate=16000)
    start = time.time()
    embedding = extractor.extract_embedding(audio, sample_rate=16000)
    elapsed = time.time() - start
    print(f"✓ Extracted in {elapsed*1000:.2f}ms")
    print(f"  Shape: {embedding.shape}")
    assert embedding.shape == (512,), f"Expected (512,), got {embedding.shape}"
    assert not np.isnan(embedding).any(), "Embedding contains NaN"

    print("\n[3/3] Testing consistency...")
    embedding2 = extractor.extract_embedding(audio, sample_rate=16000)
    max_diff = np.abs(embedding - embedding2).max()
    print(f"✓ Max difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Not consistent: {max_diff}"

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    print("\nCLAP v1.1.7 is working correctly.")
    return True


if __name__ == '__main__':
    try:
        success = quick_test()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
