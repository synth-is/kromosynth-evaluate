#!/usr/bin/env python3
"""
Manual test script for the Surrogate Quality Prediction Service.

This script tests:
1. Service connection
2. Single genome feature extraction and prediction
3. Batch prediction
4. Online training
5. Model saving/loading

Usage:
    python test_surrogate_service.py [--genome-path PATH] [--service-url URL]
    
Example:
    python test_surrogate_service.py --genome-path /path/to/genome.json
    python test_surrogate_service.py --service-url ws://localhost:32070
"""

import asyncio
import json
import argparse
import sys
import os
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import websockets
except ImportError:
    print("Error: websockets package not installed. Run: pip install websockets")
    sys.exit(1)


class SurrogateServiceTester:
    """Test client for the surrogate quality prediction service."""
    
    def __init__(self, service_url: str = "ws://localhost:32070/predict"):
        # Server accepts connections on ws://host:port (any path), but
        # default to /predict to match service banner and docs.
        self.service_url = service_url
        self.connection = None
        
    async def connect(self):
        """Establish WebSocket connection."""
        try:
            self.connection = await websockets.connect(self.service_url)
            print(f"✅ Connected to surrogate service at {self.service_url}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to {self.service_url}: {e}")
            return False
            
    async def close(self):
        """Close WebSocket connection."""
        if self.connection:
            await self.connection.close()
            print("Connection closed")
            
    async def send_request(self, request: dict) -> dict:
        """Send one request and receive one response.

        The service handles a single message per connection and then closes
        (handler returns), so we open a fresh connection per request to avoid
        1000-OK close errors when reusing the socket.
        """
        # Open a short-lived connection for this request
        # If a persistent connection is open, close it first to be safe.
        if self.connection is not None:
            try:
                await self.connection.close()
            except Exception:
                pass
            finally:
                self.connection = None

        self.connection = await websockets.connect(self.service_url)

        try:
            await self.connection.send(json.dumps(request))
            response = await self.connection.recv()
            return json.loads(response)
        finally:
            try:
                await self.connection.close()
            except Exception:
                pass
            self.connection = None
        
    async def test_status(self) -> bool:
        """Test service status."""
        print("\n📊 Testing /status endpoint...")
        try:
            response = await self.send_request({
                "type": "status"
            })
            # Server keys per ws_surrogate_service.py
            is_trained = response.get('is_trained')
            n_samples = response.get('n_training_samples')
            input_dim = response.get('input_dim')
            n_members = response.get('n_members')

            print(f"   Model trained: {is_trained}")
            print(f"   Training samples: {n_samples}")
            print(f"   Feature dimension: {input_dim}")
            print(f"   Ensemble members: {n_members}")
            return True if 'is_trained' in response else False
        except Exception as e:
            print(f"❌ Status test failed: {e}")
            return False
            
    async def test_prediction_with_genome(self, genome: dict) -> bool:
        """Test single genome prediction."""
        print("\n🔮 Testing /predict endpoint with genome...")
        try:
            response = await self.send_request({
                "genome": genome
            })
            
            if 'quality' in response and 'uncertainty' in response:
                q = response.get('quality')
                u = response.get('uncertainty')
                print(f"   Predicted quality: {q:.4f}")
                print(f"   Uncertainty: {u:.4f}")
                return True
            else:
                print(f"   Error: {response.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Prediction test failed: {e}")
            return False
            
    async def test_prediction_with_features(self, features: list) -> bool:
        """Test prediction with pre-extracted features."""
        print("\n🔮 Testing /predict endpoint with features...")
        try:
            response = await self.send_request({
                "features": features
            })
            
            if 'quality' in response and 'uncertainty' in response:
                q = response.get('quality')
                u = response.get('uncertainty')
                print(f"   Predicted quality: {q:.4f}")
                print(f"   Uncertainty: {u:.4f}")
                return True
            else:
                print(f"   Error: {response.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Prediction with features test failed: {e}")
            return False
            
    async def test_batch_prediction(self, genomes: list) -> bool:
        """Test batch prediction."""
        print(f"\n📦 Testing /predict-batch endpoint with {len(genomes)} genomes...")
        try:
            response = await self.send_request({
                "genomes": genomes
            })
            
            if 'qualities' in response and 'uncertainties' in response:
                qs = response.get('qualities', [])
                us = response.get('uncertainties', [])
                print(f"   Processed {len(qs)} genomes")
                for i in range(min(3, len(qs))):
                    print(f"   [{i}] quality: {qs[i]:.4f}, uncertainty: {us[i]:.4f}")
                if len(qs) > 3:
                    print(f"   ... and {len(qs) - 3} more")
                return True
            else:
                print(f"   Error: {response.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Batch prediction test failed: {e}")
            return False
            
    async def test_training(self, genomes: list, qualities: list) -> bool:
        """Test online training."""
        print(f"\n🎓 Testing /train endpoint with {len(genomes)} samples...")
        try:
            response = await self.send_request({
                "type": "train",
                "genomes": genomes,
                "quality_scores": qualities
            })
            
            if response.get('type') == 'train_complete':
                print(f"   Training completed!")
                # The stats keys come from surrogate_network.train_on_buffer
                # but we'll print generic presence to avoid key drift.
                for k, v in response.items():
                    if k not in ('type',):
                        print(f"   {k}: {v}")
                return True
            else:
                print(f"   Error: {response.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Training test failed: {e}")
            return False
            
    async def test_save_model(self, path: str = None) -> bool:
        """Test model saving."""
        print("\n💾 Testing /save endpoint...")
        try:
            request = {"type": "save"}
            if path:
                request["path"] = path
                
            response = await self.send_request(request)
            
            if response.get('type') == 'save_complete' or 'path' in response:
                print(f"   Model saved to: {response.get('path', 'default location')}")
                return True
            else:
                print(f"   Error: {response.get('error', 'Unknown error')}")
                return False
        except Exception as e:
            print(f"❌ Save test failed: {e}")
            return False


def create_synthetic_genome() -> dict:
    """Create a synthetic genome for testing."""
    return {
        "networkOutputs": [
            {
                "asNEATPatch": {
                    "nodes": [
                        {"activationFunction": "sine", "bias": 0.1},
                        {"activationFunction": "tanh", "bias": -0.2},
                        {"activationFunction": "relu", "bias": 0.0}
                    ],
                    "connections": [
                        {"weight": 0.5, "enabled": True},
                        {"weight": -0.3, "enabled": True}
                    ]
                }
            }
        ],
        "asNEATPatch": {
            "nodes": [
                {"type": "oscillator", "nodeType": "OscillatorNode"},
                {"type": "gain", "nodeType": "GainNode"},
                {"type": "biquadFilter", "nodeType": "BiquadFilterNode"}
            ],
            "connections": [
                {"weight": 0.8},
                {"weight": 0.6}
            ]
        }
    }


def load_genome_from_file(path: str) -> dict:
    """Load a genome from a JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


async def run_tests(args):
    """Run all tests."""
    tester = SurrogateServiceTester(args.service_url)
    
    # Connect to service
    if not await tester.connect():
        print("\n⚠️  Make sure the surrogate service is running:")
        print("   cd /path/to/kromosynth-evaluate/projection/surrogate")
        print("   ./start_surrogate_service.sh")
        return False
        
    results = {}
    
    try:
        # Test 1: Status
        results['status'] = await tester.test_status()
        
        # Load or create test genome
        if args.genome_path and os.path.exists(args.genome_path):
            print(f"\n📂 Loading genome from {args.genome_path}")
            genome = load_genome_from_file(args.genome_path)
        else:
            print("\n🔧 Using synthetic test genome")
            genome = create_synthetic_genome()
            
        # Test 2: Single prediction
        results['single_prediction'] = await tester.test_prediction_with_genome(genome)
        
        # Test 3: Prediction with synthetic features
        synthetic_features = np.random.rand(64).tolist()  # 64D feature vector
        results['features_prediction'] = await tester.test_prediction_with_features(synthetic_features)
        
        # Test 4: Batch prediction
        genomes = [create_synthetic_genome() for _ in range(5)]
        results['batch_prediction'] = await tester.test_batch_prediction(genomes)
        
        # Test 5: Training
        training_genomes = [create_synthetic_genome() for _ in range(20)]
        training_qualities = [np.random.rand() for _ in range(20)]
        results['training'] = await tester.test_training(training_genomes, training_qualities)
        
        # Test 6: Prediction after training
        results['post_training_prediction'] = await tester.test_prediction_with_genome(genome)
        
        # Test 7: Save model
        results['save_model'] = await tester.test_save_model()
        
        # Final status after training
        results['final_status'] = await tester.test_status()
        
    finally:
        await tester.close()
        
    # Summary
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"   {test_name}: {status}")
        
    print(f"\nOverall: {passed}/{total} tests passed")
    
    return all(results.values())


async def run_interactive_mode(args):
    """Run in interactive mode for manual testing."""
    tester = SurrogateServiceTester(args.service_url)
    
    if not await tester.connect():
        return
        
    print("\n🎮 Interactive mode - Enter commands (type 'help' for options, 'quit' to exit)")
    
    try:
        while True:
            cmd = input("\n> ").strip().lower()
            
            if cmd == 'quit' or cmd == 'exit':
                break
            elif cmd == 'help':
                print("""
Available commands:
  status     - Check service status
  predict    - Predict quality for synthetic genome
  batch N    - Batch predict N synthetic genomes
  train N    - Train on N synthetic samples
  save       - Save model
  quit/exit  - Exit interactive mode
                """)
            elif cmd == 'status':
                await tester.test_status()
            elif cmd == 'predict':
                await tester.test_prediction_with_genome(create_synthetic_genome())
            elif cmd.startswith('batch'):
                try:
                    n = int(cmd.split()[1]) if len(cmd.split()) > 1 else 5
                    genomes = [create_synthetic_genome() for _ in range(n)]
                    await tester.test_batch_prediction(genomes)
                except (ValueError, IndexError):
                    print("Usage: batch N (e.g., 'batch 10')")
            elif cmd.startswith('train'):
                try:
                    n = int(cmd.split()[1]) if len(cmd.split()) > 1 else 20
                    genomes = [create_synthetic_genome() for _ in range(n)]
                    qualities = [np.random.rand() for _ in range(n)]
                    await tester.test_training(genomes, qualities)
                except (ValueError, IndexError):
                    print("Usage: train N (e.g., 'train 50')")
            elif cmd == 'save':
                await tester.test_save_model()
            else:
                print(f"Unknown command: {cmd}. Type 'help' for options.")
                
    finally:
        await tester.close()


def main():
    parser = argparse.ArgumentParser(
        description="Test the Surrogate Quality Prediction Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run all tests:
    python test_surrogate_service.py
    
  Test with a specific genome file:
    python test_surrogate_service.py --genome-path /path/to/genome.json
    
  Connect to a different service URL:
    python test_surrogate_service.py --service-url ws://localhost:32071
    
  Interactive mode:
    python test_surrogate_service.py --interactive
        """
    )
    parser.add_argument(
        '--genome-path', '-g',
        type=str,
        help='Path to a genome JSON file to use for testing'
    )
    parser.add_argument(
        '--service-url', '-u',
        type=str,
        default='ws://localhost:32070',
        help='WebSocket URL of the surrogate service (default: ws://localhost:32070)'
    )
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    print("="*50)
    print("🧪 SURROGATE SERVICE TEST SUITE")
    print("="*50)
    print(f"Service URL: {args.service_url}")
    
    if args.interactive:
        asyncio.run(run_interactive_mode(args))
    else:
        success = asyncio.run(run_tests(args))
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
