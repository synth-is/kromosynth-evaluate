"""
pyribs QD Service - REST API for CMA-MAE Quality Diversity search.

Provides HTTP endpoints for ask/tell pattern with CVT-MAP-Elites archive
and CMA-MAE emitters.

Endpoints:
- POST /qd/initialize - Initialize archive and emitters
- POST /qd/ask - Get candidate solutions
- POST /qd/tell - Report evaluation results
- GET /qd/stats - Get archive statistics
- GET /qd/sample - Sample elites
- POST /qd/save - Save archive state
- POST /qd/load - Load archive state
- POST /qd/remap - Remap behavior descriptors
- GET /health - Health check
"""

from flask import Flask, request, jsonify
import numpy as np
import argparse
import os
from typing import Optional
from setproctitle import setproctitle

import sys
import os
# Add parent directory to Python path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qd.archive_manager import ArchiveManager
from qd.emitter_manager import EmitterManager
from qd.genome_codec import GenomeIDCodec, create_codec


app = Flask(__name__)

# Global state
archive_manager: Optional[ArchiveManager] = None
emitter_manager: Optional[EmitterManager] = None
genome_codec: Optional[GenomeIDCodec] = None
config: dict = {}


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    status = {
        'status': 'healthy',
        'initialized': archive_manager is not None,
        'version': '1.0.0'
    }

    if archive_manager:
        stats = archive_manager.get_stats()
        status['num_elites'] = stats['num_elites']
        status['coverage'] = stats['coverage']

    return jsonify(status), 200


@app.route('/qd/initialize', methods=['POST'])
def initialize():
    """
    Initialize or reinitialize archive and emitters.

    Request body:
    {
        "solution_dim": 1,
        "bd_dim": 6,
        "num_cells": 10000,
        "ranges": [[0, 1], [0, 1], ...],  # optional
        "num_emitters": 5,
        "sigma0": 0.5,
        "batch_size": 36,
        "seed": 42,  # optional
        "codec_type": "id",  # or "parameter"
        "genome_dir": "./genomes"  # optional
    }
    """
    global archive_manager, emitter_manager, genome_codec, config

    try:
        data = request.json
        config = data

        # Extract parameters
        solution_dim = data.get('solution_dim', 1)
        bd_dim = data.get('bd_dim', 6)
        num_cells = data.get('num_cells', 10000)
        ranges = data.get('ranges')
        if ranges:
            ranges = [tuple(r) for r in ranges]

        num_emitters = data.get('num_emitters', 5)
        sigma0 = data.get('sigma0', 0.5)
        batch_size = data.get('batch_size', 36)
        seed = data.get('seed')

        codec_type = data.get('codec_type', 'id')
        genome_dir = data.get('genome_dir', './genomes')

        # Create archive
        archive_manager = ArchiveManager(
            solution_dim=solution_dim,
            bd_dim=bd_dim,
            num_cells=num_cells,
            ranges=ranges,
            seed=seed
        )

        # Create emitters
        emitter_manager = EmitterManager(
            archive=archive_manager,
            num_emitters=num_emitters,
            sigma0=sigma0,
            batch_size=batch_size,
            seed=seed
        )

        # Create codec
        genome_codec = create_codec(
            codec_type=codec_type,
            solution_dim=solution_dim,
            genome_dir=genome_dir
        )

        return jsonify({
            'status': 'initialized',
            'config': {
                'solution_dim': solution_dim,
                'bd_dim': bd_dim,
                'num_cells': num_cells,
                'num_emitters': num_emitters,
                'batch_size': batch_size,
                'total_batch_size': num_emitters * batch_size
            }
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/qd/ask', methods=['POST'])
def ask():
    """
    Get candidate solutions from emitters.

    Request body (optional):
    {
        "count": null  # Uses default batch size if null
    }

    Response:
    {
        "solutions": [[...], [...], ...],
        "emitter_ids": [0, 0, 1, 1, ...],
        "count": 180
    }
    """
    global emitter_manager

    if emitter_manager is None:
        return jsonify({'error': 'Service not initialized. Call /qd/initialize first.'}), 400

    try:
        # Get solutions
        solutions, emitter_ids = emitter_manager.ask()

        return jsonify({
            'solutions': solutions.tolist(),
            'emitter_ids': emitter_ids,
            'count': len(solutions)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/qd/tell', methods=['POST'])
def tell():
    """
    Report evaluation results to emitters and archive.

    Request body:
    {
        "solutions": [[...], [...], ...],
        "objectives": [0.8, 0.6, ...],
        "behavior_descriptors": [[...], [...], ...],
        "metadata": [...]  # optional
    }

    Response:
    {
        "num_added": 5,
        "num_new": 3,
        "num_improved": 2,
        "indices": [...]
    }
    """
    global emitter_manager

    if emitter_manager is None:
        return jsonify({'error': 'Service not initialized. Call /qd/initialize first.'}), 400

    try:
        data = request.json

        solutions = np.array(data['solutions'])
        objectives = np.array(data['objectives'])
        behavior_descriptors = np.array(data['behavior_descriptors'])
        metadata = data.get('metadata', None)  # Optional metadata

        # Tell emitters
        stats = emitter_manager.tell(
            solutions=solutions,
            objectives=objectives,
            behavior_descriptors=behavior_descriptors,
            metadata=metadata
        )

        return jsonify(stats), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/qd/stats', methods=['GET'])
def get_stats():
    """
    Get archive statistics.

    Response:
    {
        "qd_score": 123.4,
        "coverage": 0.65,
        "max_fitness": 0.95,
        "mean_fitness": 0.68,
        "num_elites": 7321,
        ...
    }
    """
    global archive_manager

    if archive_manager is None:
        return jsonify({'error': 'Service not initialized. Call /qd/initialize first.'}), 400

    try:
        stats = archive_manager.get_stats()
        return jsonify(stats), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/qd/sample', methods=['GET'])
def sample():
    """
    Sample elites from archive.

    Query params:
    - n: number of elites to sample (default: 10)
    - return_metadata: whether to include metadata (default: false)

    Response:
    {
        "elites": [
            {
                "solution": [...],
                "objective": 0.8,
                "behavior_descriptor": [...],
                "index": 123,
                "metadata": {...}  # if return_metadata=true
            },
            ...
        ],
        "count": 10
    }
    """
    global archive_manager

    if archive_manager is None:
        return jsonify({'error': 'Service not initialized. Call /qd/initialize first.'}), 400

    try:
        n = int(request.args.get('n', 10))
        return_metadata = request.args.get('return_metadata', 'false').lower() == 'true'

        elites = archive_manager.sample_elites(n=n, return_metadata=return_metadata)

        # Convert numpy arrays to lists for JSON
        for elite in elites:
            elite['solution'] = elite['solution'].tolist()
            elite['behavior_descriptor'] = elite['behavior_descriptor'].tolist()

        return jsonify({
            'elites': elites,
            'count': len(elites)
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/qd/save', methods=['POST'])
def save():
    """
    Save archive state to disk.

    Request body:
    {
        "path": "/path/to/archive.pkl"
    }

    Response:
    {
        "status": "saved",
        "path": "/path/to/archive.pkl",
        "num_elites": 7321
    }
    """
    global archive_manager

    if archive_manager is None:
        return jsonify({'error': 'Service not initialized. Call /qd/initialize first.'}), 400

    try:
        data = request.json
        path = data['path']

        archive_manager.save(path)
        stats = archive_manager.get_stats()

        return jsonify({
            'status': 'saved',
            'path': path,
            'num_elites': stats['num_elites']
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/qd/load', methods=['POST'])
def load():
    """
    Load archive state from disk.

    Request body:
    {
        "path": "/path/to/archive.pkl"
    }

    Response:
    {
        "status": "loaded",
        "path": "/path/to/archive.pkl",
        "num_elites": 7321
    }
    """
    global archive_manager, emitter_manager

    if archive_manager is None:
        return jsonify({'error': 'Service not initialized. Call /qd/initialize first.'}), 400

    try:
        data = request.json
        path = data['path']

        archive_manager.load(path)

        # Reset emitters with loaded archive
        if emitter_manager:
            emitter_manager.reset_emitters()

        stats = archive_manager.get_stats()

        return jsonify({
            'status': 'loaded',
            'path': path,
            'num_elites': stats['num_elites']
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/qd/remap', methods=['POST'])
def remap():
    """
    Remap archive with new behavior descriptors.

    Used when projection model is updated.

    Request body:
    {
        "behavior_descriptors": [[...], [...], ...],
        "ranges": [[0, 1], [0, 1], ...]  # optional
    }

    Response:
    {
        "status": "remapped",
        "old_elites": 7321,
        "new_elites": 7103,
        "lost": 218
    }
    """
    global archive_manager, emitter_manager

    if archive_manager is None:
        return jsonify({'error': 'Service not initialized. Call /qd/initialize first.'}), 400

    try:
        data = request.json

        new_bds = np.array(data['behavior_descriptors'])
        new_ranges = data.get('ranges')
        if new_ranges:
            new_ranges = [tuple(r) for r in new_ranges]

        # Get old count
        old_stats = archive_manager.get_stats()
        old_elites = old_stats['num_elites']

        # Remap
        archive_manager.remap(new_bds=new_bds, new_ranges=new_ranges)

        # Reset emitters
        if emitter_manager:
            emitter_manager.reset_emitters()

        # Get new count
        new_stats = archive_manager.get_stats()
        new_elites = new_stats['num_elites']

        return jsonify({
            'status': 'remapped',
            'old_elites': old_elites,
            'new_elites': new_elites,
            'lost': old_elites - new_elites
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/qd/add', methods=['POST'])
def add():
    """
    Add solutions directly to archive (bypasses ask/tell protocol).

    Useful for seeding archive with initial population.

    Request body:
    {
        "solutions": [[...], [...], ...],
        "objectives": [0.8, 0.6, ...],
        "behavior_descriptors": [[...], [...], ...],
        "metadata": [...]  # optional
    }

    Response:
    {
        "num_added": 1524,
        "num_new": 1200,
        "num_improved": 324,
        "qd_score": 234.5,
        "coverage": 0.65
    }
    """
    global archive_manager

    if archive_manager is None:
        return jsonify({'error': 'Service not initialized. Call /qd/initialize first.'}), 400

    try:
        data = request.json

        solutions = np.array(data['solutions'])
        objectives = np.array(data['objectives'])
        behavior_descriptors = np.array(data['behavior_descriptors'])
        metadata = data.get('metadata', None)

        # Use ArchiveManager's add_batch method which properly handles batch operations
        add_stats = archive_manager.add_batch(
            solutions=solutions,
            objectives=objectives,
            behavior_descriptors=behavior_descriptors,
            metadata=metadata
        )

        # Get updated stats
        stats = archive_manager.get_stats()

        return jsonify({
            'num_added': add_stats['num_added'],
            'num_new': add_stats['num_new'],
            'num_improved': add_stats['num_improved'],
            'qd_score': stats['qd_score'],
            'coverage': stats['coverage']
        }), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/qd/clear', methods=['POST'])
def clear():
    """Clear archive (keeps structure, removes all elites)."""
    global archive_manager, emitter_manager

    if archive_manager is None:
        return jsonify({'error': 'Service not initialized. Call /qd/initialize first.'}), 400

    try:
        archive_manager.clear()

        # Reset emitters
        if emitter_manager:
            emitter_manager.reset_emitters()

        return jsonify({'status': 'cleared'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='pyribs QD Service - REST API for Quality Diversity search'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='Host to run server on'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=32052,
        help='Port to run server on'
    )
    parser.add_argument(
        '--process-title',
        type=str,
        default='pyribs_service',
        help='Process title'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode'
    )

    args = parser.parse_args()

    # Set process title
    if args.process_title:
        setproctitle(args.process_title)

    print("=" * 60)
    print("pyribs QD Service")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print("=" * 60)
    print("\nEndpoints:")
    print("  POST /qd/initialize - Initialize archive and emitters")
    print("  POST /qd/ask - Get candidate solutions")
    print("  POST /qd/tell - Report evaluation results")
    print("  POST /qd/add - Add solutions directly to archive (for seeding)")
    print("  GET  /qd/stats - Get archive statistics")
    print("  GET  /qd/sample?n=10 - Sample elites")
    print("  POST /qd/save - Save archive state")
    print("  POST /qd/load - Load archive state")
    print("  POST /qd/remap - Remap behavior descriptors")
    print("  GET  /health - Health check")
    print("=" * 60)
    print("\nWaiting for requests...")

    # Run Flask app
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )
