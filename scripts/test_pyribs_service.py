#!/usr/bin/env python3
"""
Test script for pyribs QD Service.

Tests the full ask/tell cycle with the REST API.
"""

import requests
import numpy as np
import time
import sys


def test_pyribs_service(base_url="http://localhost:32052"):
    """Test pyribs service endpoints."""

    print("=" * 70)
    print("pyribs QD Service Test")
    print("=" * 70)

    # Test 1: Health check
    print("\n[1/7] Testing health endpoint...")
    try:
        resp = requests.get(f"{base_url}/health")
        print(f"✓ Health check: {resp.status_code}")
        print(f"  Response: {resp.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        print("Make sure the service is running:")
        print("  ./qd/start_pyribs_service.sh")
        return False

    # Test 2: Initialize
    print("\n[2/7] Testing initialize...")
    init_config = {
        "solution_dim": 1,
        "bd_dim": 6,
        "num_cells": 100,  # Small for testing
        "num_emitters": 2,
        "sigma0": 0.5,
        "batch_size": 10,
        "seed": 42,
        "codec_type": "id",
        "genome_dir": "./test_genomes"
    }

    resp = requests.post(f"{base_url}/qd/initialize", json=init_config)
    if resp.status_code != 200:
        print(f"❌ Initialize failed: {resp.json()}")
        return False

    print(f"✓ Initialized: {resp.status_code}")
    print(f"  Config: {resp.json()['config']}")

    # Test 3: Ask
    print("\n[3/7] Testing ask...")
    resp = requests.post(f"{base_url}/qd/ask")
    if resp.status_code != 200:
        print(f"❌ Ask failed: {resp.json()}")
        return False

    ask_data = resp.json()
    solutions = np.array(ask_data['solutions'])
    emitter_ids = ask_data['emitter_ids']

    print(f"✓ Ask successful")
    print(f"  Solutions shape: {solutions.shape}")
    print(f"  Emitter IDs: {len(emitter_ids)}")
    print(f"  First 3 solutions: {solutions[:3].tolist()}")

    # Test 4: Tell
    print("\n[4/7] Testing tell...")

    # Generate random evaluations
    num_solutions = len(solutions)
    objectives = np.random.uniform(0.0, 1.0, size=num_solutions)
    behavior_descriptors = np.random.uniform(0.0, 1.0, size=(num_solutions, 6))

    tell_data = {
        "solutions": solutions.tolist(),
        "objectives": objectives.tolist(),
        "behavior_descriptors": behavior_descriptors.tolist()
    }

    resp = requests.post(f"{base_url}/qd/tell", json=tell_data)
    if resp.status_code != 200:
        print(f"❌ Tell failed: {resp.json()}")
        return False

    tell_result = resp.json()
    print(f"✓ Tell successful")
    print(f"  Added: {tell_result['num_added']}")
    print(f"  New cells: {tell_result['num_new']}")
    print(f"  Improved cells: {tell_result['num_improved']}")

    # Test 5: Stats
    print("\n[5/7] Testing stats...")
    resp = requests.get(f"{base_url}/qd/stats")
    if resp.status_code != 200:
        print(f"❌ Stats failed: {resp.json()}")
        return False

    stats = resp.json()
    print(f"✓ Stats retrieved")
    print(f"  QD Score: {stats['qd_score']:.2f}")
    print(f"  Coverage: {stats['coverage']:.2%}")
    print(f"  Num elites: {stats['num_elites']}")
    print(f"  Max fitness: {stats['max_fitness']:.3f}")

    # Test 6: Sample
    print("\n[6/7] Testing sample...")
    resp = requests.get(f"{base_url}/qd/sample?n=3")
    if resp.status_code != 200:
        print(f"❌ Sample failed: {resp.json()}")
        return False

    sample_data = resp.json()
    print(f"✓ Sample successful")
    print(f"  Sampled: {sample_data['count']} elites")
    if sample_data['count'] > 0:
        print(f"  First elite objective: {sample_data['elites'][0]['objective']:.3f}")

    # Test 7: Multiple ask/tell cycles
    print("\n[7/7] Testing multiple ask/tell cycles...")
    for i in range(3):
        # Ask
        resp = requests.post(f"{base_url}/qd/ask")
        solutions = np.array(resp.json()['solutions'])

        # Evaluate (random)
        objectives = np.random.uniform(0.0, 1.0, size=len(solutions))
        bds = np.random.uniform(0.0, 1.0, size=(len(solutions), 6))

        # Tell
        resp = requests.post(f"{base_url}/qd/tell", json={
            "solutions": solutions.tolist(),
            "objectives": objectives.tolist(),
            "behavior_descriptors": bds.tolist()
        })

        result = resp.json()
        print(f"  Cycle {i+1}: Added {result['num_added']}, "
              f"New {result['num_new']}, Improved {result['num_improved']}")

    # Final stats
    print("\n" + "=" * 70)
    print("Final Stats:")
    print("=" * 70)
    resp = requests.get(f"{base_url}/qd/stats")
    stats = resp.json()
    print(f"QD Score: {stats['qd_score']:.2f}")
    print(f"Coverage: {stats['coverage']:.2%} ({stats['num_elites']}/{stats['cells_total']} cells)")
    print(f"Max fitness: {stats['max_fitness']:.3f}")
    print(f"Mean fitness: {stats['mean_fitness']:.3f}")

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)

    return True


if __name__ == '__main__':
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:32052"

    print(f"\nTesting service at: {base_url}")
    print("Make sure the service is running first!")
    print()

    success = test_pyribs_service(base_url)
    sys.exit(0 if success else 1)
