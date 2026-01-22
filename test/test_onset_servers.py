import asyncio
import websockets
import json
import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_pipeline(audio_path, onset_port=8080, discr_port=8081, fitness_port=8082,
                       n_bins=8, n_intensity_levels=6, novelty_weight=0.5):
    """Test the complete pipeline: onset detection → discretization → fitness evaluation."""
    try:
        # Load audio file
        logger.info(f"Loading audio file: {audio_path}")
        audio_data, sr = librosa.load(audio_path, sr=None)
        logger.info(f"Audio loaded: {len(audio_data)/sr:.2f} seconds, {sr} Hz sample rate")
        
        # Convert to float32 and ensure mono
        audio_data = audio_data.astype(np.float32)
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # 1. Onset Detection
        onset_uri = f"ws://localhost:{onset_port}?n_bins={n_bins}&sample_rate={sr}"
        async with websockets.connect(onset_uri) as websocket:
            logger.info("\nConnected to onset detection server")
            logger.info("Sending audio data for onset detection")
            
            await websocket.send(audio_data.tobytes())
            response = await websocket.recv()
            onset_data = json.loads(response)
            
            if onset_data['status'] != 'OK':
                logger.error(f"Onset detection error: {onset_data['status']}")
                return
            
            features = onset_data['features']
            logger.info("\nOnset Detection Results:")
            logger.info(f"Number of onsets detected: {len(features['onset_times'])}")
            logger.info(f"Tempo: {features['tempo']:.1f} BPM")
            logger.info(f"Bin edges: {[f'{t:.2f}s' for t in features['bin_edges']]}")
            logger.info(f"Max strengths per bin: {[f'{s:.3f}' for s in features['max_strengths_per_bin']]}")
        
        # 2. Discretization
        discr_uri = f"ws://localhost:{discr_port}?n_intensity_levels={n_intensity_levels}"
        async with websockets.connect(discr_uri) as websocket:
            logger.info("\nConnected to discretization server")
            logger.info("Sending onset features for discretization")
            
            feature_vectors = [features]  # Wrap features in array
            data = {'feature_vectors': feature_vectors}
            
            await websocket.send(json.dumps(data))
            response = await websocket.recv()
            discr_data = json.loads(response)
            
            if discr_data['status'] != 'OK':
                logger.error(f"Discretization error: {discr_data['status']}")
                return
            
            logger.info("\nDiscretization Results:")
            logger.info(f"Discretized vector: {discr_data['feature_map'][0]}")
        
        # 3. Fitness Evaluation
        fitness_base_uri = f"ws://localhost:{fitness_port}"
        
        # Get conventional fitness
        async with websockets.connect(f"{fitness_base_uri}/conventional") as websocket:
            logger.info("\nConnected to fitness server (conventional)")
            await websocket.send(json.dumps(features))
            response = await websocket.recv()
            conv_data = json.loads(response)
            
        # Get novelty fitness
        async with websockets.connect(f"{fitness_base_uri}/novelty") as websocket:
            logger.info("Connected to fitness server (novelty)")
            await websocket.send(json.dumps(features))
            response = await websocket.recv()
            nov_data = json.loads(response)
            
        # Get hybrid fitness
        async with websockets.connect(f"{fitness_base_uri}/hybrid?novelty_weight={novelty_weight}") as websocket:
            logger.info("Connected to fitness server (hybrid)")
            await websocket.send(json.dumps(features))
            response = await websocket.recv()
            hybrid_data = json.loads(response)
        
        logger.info("\nFitness Results:")
        logger.info(f"Conventional fitness: {conv_data['fitness']:.3f}")
        logger.info("Conventional components:")
        for name, value in conv_data['components'].items():
            logger.info(f"  {name}: {value:.3f}")
            
        logger.info(f"\nNovelty fitness: {nov_data['fitness']:.3f}")
        logger.info("Novelty components:")
        for name, value in nov_data['components'].items():
            logger.info(f"  {name}: {value:.3f}")
            
        logger.info(f"\nHybrid fitness (novelty_weight={novelty_weight}): {hybrid_data['fitness']:.3f}")
        
        # Create visualization
        logger.info("\nCreating visualization...")
        fig = plt.figure(figsize=(15, 12))
        
        # Plot waveform
        plt.subplot(4, 1, 1)
        times = np.arange(len(audio_data)) / sr
        plt.plot(times, audio_data)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot onset strengths
        plt.subplot(4, 1, 2)
        onset_times = np.array(features['onset_times'])
        onset_strengths = np.array(features['onset_strengths'])
        plt.vlines(onset_times, 0, onset_strengths, color='r', alpha=0.7, label='Onsets')
        for edge in features['bin_edges']:
            plt.axvline(edge, color='g', alpha=0.3, linestyle='--')
        plt.title('Onset Strengths and Bin Edges')
        plt.xlabel('Time (s)')
        plt.ylabel('Strength')
        plt.legend()
        
        # Plot discretized vector
        plt.subplot(4, 1, 3)
        plt.bar(range(len(discr_data['feature_map'][0])), 
                discr_data['feature_map'][0],
                tick_label=[f'Bin {i+1}' for i in range(len(discr_data['feature_map'][0]))])
        plt.title('Discretized Onset Pattern')
        plt.xlabel('Bin')
        plt.ylabel('Intensity Level')
        
        # Plot fitness components
        plt.subplot(4, 1, 4)
        all_components = {
            'Conv: ' + k: v for k, v in conv_data['components'].items()
        }
        all_components.update({
            'Nov: ' + k: v for k, v in nov_data['components'].items()
        })
        
        names = list(all_components.keys())
        values = list(all_components.values())
        
        plt.bar(range(len(values)), values, tick_label=names)
        plt.title('Fitness Components')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Score')
        
        plt.tight_layout()
        
        # Save results
        output_dir = Path('onset_test_results')
        output_dir.mkdir(exist_ok=True)
        
        # Save plot
        plot_path = output_dir / f"{Path(audio_path).stem}_analysis.png"
        plt.savefig(plot_path)
        logger.info(f"Visualization saved to: {plot_path}")
        
        # Save detailed results
        results = {
            'audio_file': str(audio_path),
            'duration': len(audio_data)/sr,
            'sample_rate': sr,
            'tempo': features['tempo'],
            'n_onsets': len(features['onset_times']),
            'onset_times': features['onset_times'],
            'onset_strengths': features['onset_strengths'],
            'bin_edges': features['bin_edges'],
            'max_strengths_per_bin': features['max_strengths_per_bin'],
            'discretized_vector': discr_data['feature_map'][0],
            'fitness': {
                'conventional': {
                    'total': conv_data['fitness'],
                    'components': conv_data['components']
                },
                'novelty': {
                    'total': nov_data['fitness'],
                    'components': nov_data['components']
                },
                'hybrid': {
                    'total': hybrid_data['fitness'],
                    'components': hybrid_data['components']
                }
            }
        }
        
        json_path = output_dir / f"{Path(audio_path).stem}_results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Detailed results saved to: {json_path}")

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test onset detection pipeline.')
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    parser.add_argument('--onset-port', type=int, default=8080, help='Port for onset detection server')
    parser.add_argument('--discr-port', type=int, default=8081, help='Port for discretization server')
    parser.add_argument('--fitness-port', type=int, default=8082, help='Port for fitness server')
    parser.add_argument('--n-bins', type=int, default=8, help='Number of temporal bins')
    parser.add_argument('--n-intensity-levels', type=int, default=6, 
                       help='Number of intensity levels for discretization')
    parser.add_argument('--novelty-weight', type=float, default=0.5,
                       help='Weight for novelty in hybrid fitness (0.0-1.0)')
    
    args = parser.parse_args()
    
    asyncio.run(test_pipeline(
        args.audio_path,
        onset_port=args.onset_port,
        discr_port=args.discr_port,
        fitness_port=args.fitness_port,
        n_bins=args.n_bins,
        n_intensity_levels=args.n_intensity_levels,
        novelty_weight=args.novelty_weight
    ))