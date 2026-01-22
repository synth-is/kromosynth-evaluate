#!/usr/bin/env python3
"""
Test client for QDHF projection WebSocket service.

Tests single and batch projection inference.

Usage:
    # Start service first (terminal 1):
    ./projection/qdhf/start_projection_service.sh --model models/projection/projection_v1.pt

    # Run tests (terminal 2):
    python scripts/test_projection_service.py
"""

import asyncio
import websockets
import json
import numpy as np
import sys
from pathlib import Path


async def test_single_projection(uri: str):
    """Test single embedding projection."""
    print("Testing single projection...")

    # Generate random CLAP embedding (512D)
    embedding = np.random.randn(512).astype(np.float32)

    # Create request
    request = {
        'embedding': embedding.tolist(),
        'sound_id': 'test_sound_001'
    }

    # Send request
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(request))

        # Receive response
        response = await websocket.recv()
        data = json.loads(response)

        # Validate response
        assert 'behavior_descriptor' in data, "Missing behavior_descriptor in response"
        assert 'sound_id' in data, "Missing sound_id in response"
        assert 'inference_time_ms' in data, "Missing inference_time_ms in response"

        bd = np.array(data['behavior_descriptor'])
        assert bd.shape == (6,), f"Invalid BD shape: {bd.shape}"
        assert np.all((bd >= 0) & (bd <= 1)), "BD values not in [0, 1] range"

        print(f"  ✓ Single projection successful")
        print(f"  BD: {bd}")
        print(f"  Inference time: {data['inference_time_ms']:.2f}ms")
        print()


async def test_batch_projection(uri: str, batch_size: int = 10):
    """Test batch embedding projection."""
    print(f"Testing batch projection (size={batch_size})...")

    # Generate random CLAP embeddings (N, 512)
    embeddings = np.random.randn(batch_size, 512).astype(np.float32)
    sound_ids = [f'test_sound_{i:03d}' for i in range(batch_size)]

    # Create request
    request = {
        'embeddings': embeddings.tolist(),
        'sound_ids': sound_ids
    }

    # Send request
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps(request))

        # Receive response
        response = await websocket.recv()
        data = json.loads(response)

        # Validate response
        assert 'behavior_descriptors' in data, "Missing behavior_descriptors in response"
        assert 'sound_ids' in data, "Missing sound_ids in response"
        assert 'count' in data, "Missing count in response"
        assert 'inference_time_ms' in data, "Missing inference_time_ms in response"

        bds = np.array(data['behavior_descriptors'])
        assert bds.shape == (batch_size, 6), f"Invalid BDs shape: {bds.shape}"
        assert np.all((bds >= 0) & (bds <= 1)), "BD values not in [0, 1] range"
        assert data['count'] == batch_size, "Count mismatch"
        assert data['sound_ids'] == sound_ids, "Sound IDs mismatch"

        print(f"  ✓ Batch projection successful")
        print(f"  Shape: {bds.shape}")
        print(f"  Inference time: {data['inference_time_ms']:.2f}ms")
        print(f"  Time per sample: {data['inference_time_ms'] / batch_size:.2f}ms")
        print(f"  Sample BDs:")
        for i in range(min(3, batch_size)):
            print(f"    {sound_ids[i]}: {bds[i]}")
        print()


async def test_error_handling(uri: str):
    """Test error handling."""
    print("Testing error handling...")

    # Test 1: Invalid JSON
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send("invalid json {")
            response = await websocket.recv()
            data = json.loads(response)
            assert 'error' in data, "Expected error in response"
            print(f"  ✓ Invalid JSON handled correctly")
    except Exception as e:
        print(f"  ✗ Error handling failed: {e}")

    # Test 2: Wrong embedding dimension
    try:
        async with websockets.connect(uri) as websocket:
            request = {
                'embedding': np.random.randn(256).tolist()  # Wrong dimension
            }
            await websocket.send(json.dumps(request))
            response = await websocket.recv()
            data = json.loads(response)
            assert 'error' in data, "Expected error in response"
            print(f"  ✓ Wrong dimension handled correctly")
    except Exception as e:
        print(f"  ✗ Error handling failed: {e}")

    print()


async def test_performance(uri: str, num_requests: int = 100):
    """Test service performance."""
    print(f"Testing performance ({num_requests} requests)...")

    import time

    times = []

    for i in range(num_requests):
        embedding = np.random.randn(512).astype(np.float32)

        request = {
            'embedding': embedding.tolist()
        }

        start = time.time()

        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(request))
            response = await websocket.recv()

        elapsed = (time.time() - start) * 1000
        times.append(elapsed)

    times = np.array(times)

    print(f"  ✓ Performance test complete")
    print(f"  Requests: {num_requests}")
    print(f"  Mean time: {times.mean():.2f}ms")
    print(f"  Std time: {times.std():.2f}ms")
    print(f"  Min time: {times.min():.2f}ms")
    print(f"  Max time: {times.max():.2f}ms")
    print(f"  p50: {np.percentile(times, 50):.2f}ms")
    print(f"  p95: {np.percentile(times, 95):.2f}ms")
    print(f"  p99: {np.percentile(times, 99):.2f}ms")
    print()


async def main():
    uri = "ws://localhost:32053/project"

    print("=" * 80)
    print("QDHF Projection Service Test Client")
    print("=" * 80)
    print()

    try:
        # Basic tests
        await test_single_projection(uri)
        await test_batch_projection(uri, batch_size=10)
        await test_batch_projection(uri, batch_size=100)

        # Error handling
        await test_error_handling(uri)

        # Performance
        await test_performance(uri, num_requests=100)

        print("=" * 80)
        print("All tests passed!")
        print("=" * 80)

    except websockets.exceptions.WebSocketException as e:
        print(f"Connection error: {e}")
        print()
        print("Make sure the projection service is running:")
        print("  ./projection/qdhf/start_projection_service.sh --model models/projection/projection_v1.pt")
        sys.exit(1)

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    asyncio.run(main())
