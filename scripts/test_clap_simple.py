#!/usr/bin/env python3
"""
Simple standalone test for CLAP extractor.
Tests basic functionality without requiring pytest.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from features.clap.clap_extractor import CLAPExtractor


def generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000):
    """Generate a simple sine wave for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


def test_clap_extractor():
    """Run basic tests on CLAP extractor."""

    print("=" * 70)
    print("CLAP Extractor Test Suite")
    print("=" * 70)

    # Test 1: Initialize extractor
    print("\n[1/7] Initializing CLAP extractor...")
    start = time.time()
    extractor = CLAPExtractor(device='cpu')  # Use CPU for testing
    elapsed = time.time() - start
    print(f"✓ Extractor initialized in {elapsed:.2f}s")
    print(f"  Device: {extractor.device}")
    print(f"  Target sample rate: {extractor.target_sample_rate} Hz")

    # Test 2: Single embedding extraction
    print("\n[2/7] Testing single embedding extraction...")
    audio = generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000)
    start = time.time()
    embedding = extractor.extract_embedding(audio, sample_rate=16000)
    elapsed = time.time() - start
    print(f"✓ Extracted embedding in {elapsed*1000:.2f}ms")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding dtype: {embedding.dtype}")
    print(f"  Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")

    assert embedding.shape == (512,), f"Expected (512,), got {embedding.shape}"
    assert not np.isnan(embedding).any(), "Embedding contains NaN"
    assert not np.isinf(embedding).any(), "Embedding contains Inf"

    # Test 3: Embedding consistency
    print("\n[3/7] Testing embedding consistency...")
    embedding2 = extractor.extract_embedding(audio, sample_rate=16000)
    max_diff = np.abs(embedding - embedding2).max()
    print(f"✓ Maximum difference: {max_diff:.2e}")
    assert max_diff < 1e-5, f"Embeddings not consistent, max diff: {max_diff}"

    # Test 4: Different frequencies
    print("\n[4/7] Testing different frequencies produce different embeddings...")
    audio_440 = generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000)
    audio_880 = generate_sine_wave(frequency=880, duration=1.0, sample_rate=16000)

    emb_440 = extractor.extract_embedding(audio_440, sample_rate=16000)
    emb_880 = extractor.extract_embedding(audio_880, sample_rate=16000)

    similarity = extractor.compute_similarity(emb_440, emb_880)
    distance = extractor.compute_distance(emb_440, emb_880)

    print(f"  440Hz vs 880Hz similarity: {similarity:.4f}")
    print(f"  440Hz vs 880Hz distance: {distance:.4f}")

    assert similarity < 1.0, "Different frequencies should not be identical"
    assert np.isclose(distance, 1.0 - similarity), "Distance should equal 1 - similarity"
    print("✓ Different frequencies produce different embeddings")

    # Test 5: Batch extraction
    print("\n[5/7] Testing batch extraction...")
    audio_buffers = [
        generate_sine_wave(frequency=220, duration=1.0, sample_rate=16000),
        generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000),
        generate_sine_wave(frequency=880, duration=1.0, sample_rate=16000),
    ]

    start = time.time()
    embeddings = extractor.extract_batch(audio_buffers, sample_rate=16000)
    elapsed = time.time() - start

    print(f"✓ Extracted {len(audio_buffers)} embeddings in {elapsed*1000:.2f}ms")
    print(f"  Avg per embedding: {elapsed*1000/len(audio_buffers):.2f}ms")
    print(f"  Batch shape: {embeddings.shape}")

    assert embeddings.shape == (3, 512), f"Expected (3, 512), got {embeddings.shape}"

    # Test 6: Silent audio
    print("\n[6/7] Testing silent audio handling...")
    silent = np.zeros(16000, dtype=np.float32)
    emb_silent = extractor.extract_embedding(silent, sample_rate=16000)
    print(f"✓ Silent audio handled successfully")
    print(f"  Embedding shape: {emb_silent.shape}")
    assert not np.isnan(emb_silent).any(), "Silent audio produced NaN"

    # Test 7: Sample rate resampling
    print("\n[7/7] Testing sample rate resampling...")
    audio_44k = generate_sine_wave(frequency=440, duration=1.0, sample_rate=44100)
    emb_44k = extractor.extract_embedding(audio_44k, sample_rate=44100)

    # Compare with 16kHz version
    audio_16k = generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000)
    emb_16k = extractor.extract_embedding(audio_16k, sample_rate=16000)

    sim_resample = extractor.compute_similarity(emb_44k, emb_16k)
    print(f"  44.1kHz vs 16kHz similarity: {sim_resample:.4f}")
    print(f"✓ Resampling works correctly")

    assert sim_resample > 0.8, f"Resampled audio should be similar, got {sim_resample}"

    # Summary
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)
    print("\nCLAP extractor is working correctly and ready for use.")
    print("\nPerformance summary:")
    print(f"  - Single extraction: ~{elapsed*1000/len(audio_buffers):.0f}ms per audio (CPU)")
    print(f"  - Embedding dimension: 512")
    print(f"  - Target sample rate: 48kHz (auto-resamples)")
    print(f"  - Deterministic: Yes (same audio → same embedding)")

    return True


if __name__ == '__main__':
    try:
        success = test_clap_extractor()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
