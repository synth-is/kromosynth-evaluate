"""
Unit tests for CLAP feature extractor.

Tests:
- Initialization
- Single embedding extraction
- Batch embedding extraction
- Embedding consistency (same audio → same embedding)
- Edge cases (silent audio, very short audio)
- Similarity computation
"""

import sys
sys.path.append('..')

import numpy as np
import pytest
from features.clap.clap_extractor import CLAPExtractor


@pytest.fixture
def extractor():
    """Create CLAP extractor instance (reused across tests)."""
    return CLAPExtractor(device='cpu')  # Use CPU for testing


def generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000):
    """Generate a simple sine wave for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio


def test_extractor_initialization(extractor):
    """Test that extractor initializes correctly."""
    assert extractor is not None
    assert extractor.target_sample_rate == 48000
    assert extractor.device in ['cuda', 'cpu']


def test_single_embedding_extraction(extractor):
    """Test extracting embedding from single audio buffer."""
    # Generate test audio
    audio = generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000)

    # Extract embedding
    embedding = extractor.extract_embedding(audio, sample_rate=16000)

    # Check shape and type
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,), f"Expected (512,), got {embedding.shape}"
    assert embedding.dtype in [np.float32, np.float64]

    # Check values are normalized (typical for CLAP embeddings)
    assert not np.isnan(embedding).any(), "Embedding contains NaN values"
    assert not np.isinf(embedding).any(), "Embedding contains Inf values"


def test_batch_embedding_extraction(extractor):
    """Test batch extraction of embeddings."""
    # Generate multiple test audio buffers
    audio_buffers = [
        generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000),
        generate_sine_wave(frequency=880, duration=1.0, sample_rate=16000),
        generate_sine_wave(frequency=220, duration=1.0, sample_rate=16000),
    ]

    # Extract embeddings
    embeddings = extractor.extract_batch(audio_buffers, sample_rate=16000)

    # Check shape
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, 512), f"Expected (3, 512), got {embeddings.shape}"

    # Check no NaN or Inf
    assert not np.isnan(embeddings).any()
    assert not np.isinf(embeddings).any()


def test_embedding_consistency(extractor):
    """Test that same audio produces same embedding."""
    audio = generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000)

    # Extract embedding twice
    embedding1 = extractor.extract_embedding(audio, sample_rate=16000)
    embedding2 = extractor.extract_embedding(audio, sample_rate=16000)

    # Check they are identical (or very close due to floating point)
    np.testing.assert_allclose(
        embedding1,
        embedding2,
        rtol=1e-5,
        atol=1e-7,
        err_msg="Same audio should produce same embedding"
    )


def test_different_frequencies_produce_different_embeddings(extractor):
    """Test that different audio produces different embeddings."""
    audio1 = generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000)
    audio2 = generate_sine_wave(frequency=880, duration=1.0, sample_rate=16000)

    embedding1 = extractor.extract_embedding(audio1, sample_rate=16000)
    embedding2 = extractor.extract_embedding(audio2, sample_rate=16000)

    # Check they are different
    assert not np.allclose(embedding1, embedding2), \
        "Different audio should produce different embeddings"


def test_silent_audio(extractor):
    """Test handling of silent audio."""
    # Generate silent audio
    audio = np.zeros(16000, dtype=np.float32)

    # Should not crash, should return valid embedding
    embedding = extractor.extract_embedding(audio, sample_rate=16000)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)
    assert not np.isnan(embedding).any()


def test_very_short_audio(extractor):
    """Test handling of very short audio."""
    # Generate very short audio (0.1 seconds)
    audio = generate_sine_wave(frequency=440, duration=0.1, sample_rate=16000)

    # Should not crash
    embedding = extractor.extract_embedding(audio, sample_rate=16000)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (512,)


def test_similarity_computation(extractor):
    """Test similarity computation between embeddings."""
    audio1 = generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000)
    audio2 = generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000)
    audio3 = generate_sine_wave(frequency=880, duration=1.0, sample_rate=16000)

    embedding1 = extractor.extract_embedding(audio1, sample_rate=16000)
    embedding2 = extractor.extract_embedding(audio2, sample_rate=16000)
    embedding3 = extractor.extract_embedding(audio3, sample_rate=16000)

    # Same audio should have similarity ≈ 1.0
    sim_same = extractor.compute_similarity(embedding1, embedding2)
    assert 0.99 <= sim_same <= 1.0, f"Expected ~1.0, got {sim_same}"

    # Different audio should have lower similarity
    sim_different = extractor.compute_similarity(embedding1, embedding3)
    assert 0.0 <= sim_different <= 1.0, f"Similarity out of range: {sim_different}"
    assert sim_different < sim_same, "Different audio should have lower similarity"


def test_distance_computation(extractor):
    """Test distance computation between embeddings."""
    audio1 = generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000)
    audio2 = generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000)

    embedding1 = extractor.extract_embedding(audio1, sample_rate=16000)
    embedding2 = extractor.extract_embedding(audio2, sample_rate=16000)

    # Distance should be 1 - similarity
    distance = extractor.compute_distance(embedding1, embedding2)
    similarity = extractor.compute_similarity(embedding1, embedding2)

    np.testing.assert_allclose(
        distance,
        1.0 - similarity,
        rtol=1e-5,
        err_msg="Distance should equal 1 - similarity"
    )


def test_sample_rate_resampling(extractor):
    """Test that different sample rates are handled correctly."""
    # Generate audio at different sample rates
    audio_16k = generate_sine_wave(frequency=440, duration=1.0, sample_rate=16000)
    audio_44k = generate_sine_wave(frequency=440, duration=1.0, sample_rate=44100)

    # Extract embeddings
    embedding_16k = extractor.extract_embedding(audio_16k, sample_rate=16000)
    embedding_44k = extractor.extract_embedding(audio_44k, sample_rate=44100)

    # Both should produce valid embeddings
    assert embedding_16k.shape == (512,)
    assert embedding_44k.shape == (512,)

    # They should be similar (same frequency content)
    similarity = extractor.compute_similarity(embedding_16k, embedding_44k)
    assert similarity > 0.8, \
        f"Same audio at different sample rates should be similar, got {similarity}"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
