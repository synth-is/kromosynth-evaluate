#!/usr/bin/env python3
"""
Test that CLAP module imports correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing CLAP module imports...")

try:
    print("  Importing laion_clap...")
    import laion_clap
    print("  ✓ laion_clap imported")

    print("  Importing torch...")
    import torch
    print(f"  ✓ torch imported (version {torch.__version__})")

    print("  Importing CLAPExtractor...")
    from features.clap.clap_extractor import CLAPExtractor
    print("  ✓ CLAPExtractor imported")

    print("\nAll imports successful!")
    print("\nNote: To test CLAP extraction, run test_clap_simple.py")
    print("(This will download the CLAP checkpoint on first run, ~500MB)")

except ImportError as e:
    print(f"\n❌ Import failed: {e}")
    sys.exit(1)
