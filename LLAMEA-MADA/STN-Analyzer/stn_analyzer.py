#!/usr/bin/env python3
"""
STN Analyzer for MADA-LLAMEA Experiments

A standalone tool for analyzing and visualizing Search Trajectory Networks
from MADA-LLAMEA experiment results.

This tool works with existing experiment data without requiring new experiments
or modifications to the main MADA-LLAMEA codebase.

Features:
- Load and parse mada_offspring.jsonl for lineage information
- Build NetworkX graphs representing search trajectories
- Calculate comprehensive STN metrics
- Generate static and interactive visualizations
- Support batch analysis of multiple experiments

Usage:
    # Analyze single experiment
    python stn_analyzer.py --exp-dir path/to/experiment
    
    # Analyze with interactive HTML output
    python stn_analyzer.py --exp-dir path/to/experiment --interactive
    
    # Batch analyze all experiments in a folder
    python stn_analyzer.py --batch path/to/experiments/folder
    
    # Specify custom output directory
    python stn_analyzer.py --exp-dir exp-folder --output-dir my_outputs

Author: MADA-LLAMEA Team
Version: 1.0.0
"""

import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from tqdm import tqdm

# Fix Windows console encoding for Unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7 fallback
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add parent directory to path for imports when run as script
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))

from stn.data_loader import ExperimentDataLoader
from stn.graph_builder import STNGraphBuilder
from stn.metrics import STNMetrics
from stn.visualizer import STNVisualizer


def analyze_experiment(
    exp_dir: Path,
    output_dir: Path,
    interactive: bool = False,
    verbose: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Analyze a single experiment and generate STN visualizations.
    
    Args:
        exp_dir: Path to experiment directory
        output_dir: Base output directory for results
        interactive: Generate interactive HTML visualization
        verbose: Print progress messages
        
    Returns:
        Dictionary with all metrics, or None if analysis failed
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Analyzing: {exp_dir.name}")
        print('='*60)
    
    # Create progress bar for main steps
    total_steps = 6  # Load, Build, Metrics, Visualizations, Save metrics, Save lineages
    with tqdm(total=total_steps, desc="Analysis Progress", unit="step", disable=not verbose) as main_pbar:
        
        # Step 1: Load data
        main_pbar.set_description("Step 1/6: Loading data")
        if verbose:
            print(f"\n[Step 1/6] Loading experiment data from: {exp_dir}")
        
        loader = ExperimentDataLoader(exp_dir)
        try:
            if verbose:
                print(f"  Starting data load...")
            nodes, edges = loader.load()
            if verbose:
                print(f"[OK] Loaded {len(nodes)} nodes and {len(edges)} edges")
        except FileNotFoundError as e:
            if verbose:
                print(f"[ERROR] Skipping: {e}")
                import traceback
                traceback.print_exc()
            return None
        except Exception as e:
            if verbose:
                print(f"[ERROR] Error loading data: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            return None
        main_pbar.update(1)
        
        if not nodes:
            if verbose:
                print("[ERROR] No valid data found")
            return None
        
        # Step 2: Build graph
        main_pbar.set_description("Step 2/6: Building graph")
        if verbose:
            print(f"\n[Step 2/6] Building STN graph from {len(nodes)} nodes and {len(edges)} edges")
        
        try:
            builder = STNGraphBuilder()
            if verbose:
                print(f"  Building graph structure...")
            graph = builder.build_graph(nodes, edges)
            if verbose:
                print(f"[OK] Built STN graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to build graph: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            return None
        main_pbar.update(1)
        
        # Step 3: Calculate metrics
        main_pbar.set_description("Step 3/6: Calculating metrics")
        if verbose:
            print(f"\n[Step 3/6] Calculating STN metrics")
        
        try:
            metrics = STNMetrics(graph)
            
            if verbose:
                print(f"  Computing basic metrics...")
            basic = metrics.basic_metrics()
            
            if verbose:
                print(f"  Computing operator analysis...")
            operator_stats = metrics.operator_analysis()
            
            if verbose:
                print(f"  Computing trajectory analysis...")
            trajectory_stats = metrics.trajectory_analysis()
            
            if verbose:
                print(f"  Computing fitness landscape analysis...")
            fitness_stats = metrics.fitness_landscape_analysis()
            
            if verbose:
                print(f"  Computing diversity analysis...")
            diversity_stats = metrics.diversity_analysis()
            
            if verbose:
                print(f"  Computing bandit analysis...")
            bandit_stats = metrics.bandit_analysis()
            
            if verbose:
                print(f"  Identifying successful lineages...")
            lineages = metrics.identify_successful_lineages()
            
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to calculate metrics: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            return None
        
        # Print summary
        if verbose:
            print(f"\n--- Basic Metrics ---")
            print(f"  Nodes: {basic['num_nodes']}")
            print(f"  Edges: {basic['num_edges']}")
            print(f"  DAG: {basic['is_dag']}")
            print(f"  Connected Components: {basic['num_connected_components']}")
            print(f"  Max Depth: {basic['max_depth']}")
            
            print(f"\n--- Operator Analysis ---")
            print(f"  Counts: {operator_stats['operator_counts']}")
            print(f"  Most Effective: {operator_stats['most_effective']}")
            for op, stats in operator_stats['per_operator'].items():
                if stats['count'] > 0:
                    print(f"    {op}: {stats['count']} uses, "
                          f"{stats['success_rate']:.1%} success, "
                          f"μ_imp={stats['mean_improvement']:.4f}")
            
            print(f"\n--- Trajectory Analysis ---")
            print(f"  Roots: {trajectory_stats['num_roots']}")
            print(f"  Leaves: {trajectory_stats['num_leaves']}")
            print(f"  Avg Path Length: {trajectory_stats['avg_path_length']:.2f}")
            print(f"  Branching Factor: {trajectory_stats['branching_factor']:.2f}")
            
            print(f"\n--- Fitness Analysis ---")
            print(f"  Best: {fitness_stats['best_node']} (fitness: {fitness_stats['fitness_max']:.4f})")
            print(f"  Mean Fitness: {fitness_stats['fitness_mean']:.4f}")
            print(f"  Improvement Trend: {fitness_stats['fitness_improvement_trend']}")
            print(f"  Error Rate: {fitness_stats['error_rate']:.1%}")
            print(f"  Found {len(lineages)} successful lineages")
            
            if diversity_stats['mean_nn_dist'] > 0:
                print(f"\n--- Diversity Analysis ---")
                print(f"  Mean NN-Dist: {diversity_stats['mean_nn_dist']:.4f}")
                print(f"  Mean Alpha: {diversity_stats['mean_alpha']:.4f}")
        main_pbar.update(1)
        
        # Step 4: Create visualizations
        main_pbar.set_description("Step 4/6: Creating visualizations")
        if verbose:
            print(f"\n[Step 4/6] Creating visualizations")
        
        try:
            exp_output = output_dir / exp_dir.name
            exp_output.mkdir(parents=True, exist_ok=True)
            
            if verbose:
                print(f"  Output directory: {exp_output}")
            
            viz = STNVisualizer(graph, exp_output)
            
            viz_outputs = viz.generate_all_visualizations(
                lineages=lineages,
                interactive=interactive,
            )
            
            if verbose:
                print(f"[OK] Generated {len([v for v in viz_outputs.values() if v])} visualizations")
        except Exception as e:
            if verbose:
                print(f"[WARNING] Error creating visualizations: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
            # Continue even if visualization fails
            viz_outputs = {}
        main_pbar.update(1)
        
        # Step 5: Save metrics to JSON
        main_pbar.set_description("Step 5/6: Saving metrics")
        if verbose:
            print(f"\n[Step 5/6] Saving metrics to JSON")
        
        try:
            all_metrics = {
                'experiment_name': exp_dir.name,
                'analysis_timestamp': datetime.now().isoformat(),
                'summary': loader.get_experiment_summary(),
                'basic': basic,
                'operator': operator_stats,
                'trajectory': trajectory_stats,
                'fitness': {k: v for k, v in fitness_stats.items() 
                           if k not in ['generation_stats']},  # Exclude large nested dict
                'diversity': diversity_stats,
                'bandit': bandit_stats,
                'best_node': fitness_stats.get('best_node'),
                'num_successful_lineages': len(lineages),
                'visualization_files': {k: str(v) if v else None for k, v in viz_outputs.items()},
            }
            
            # Add generation stats separately (can be large)
            gen_stats_file = exp_output / "generation_stats.json"
            if verbose:
                print(f"  Saving generation stats...")
            with open(gen_stats_file, 'w', encoding='utf-8') as f:
                json.dump(fitness_stats.get('generation_stats', {}), f, indent=2, default=str)
            if verbose:
                print(f"[OK] Saved: {gen_stats_file.name}")
            
            # Save main metrics
            metrics_file = exp_output / "stn_metrics.json"
            if verbose:
                print(f"  Saving main metrics...")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(all_metrics, f, indent=2, default=str)
            if verbose:
                print(f"[OK] Saved: {metrics_file.name}")
                
        except Exception as e:
            if verbose:
                print(f"[WARNING] Error saving metrics: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        
        main_pbar.update(1)
        
        # Step 6: Save lineages
        main_pbar.set_description("Step 6/6: Saving lineages")
        if verbose:
            print(f"\n[Step 6/6] Saving lineages and graph data")
        
        try:
            if lineages:
                lineages_file = exp_output / "successful_lineages.json"
                if verbose:
                    print(f"  Saving {len(lineages)} lineages...")
                with open(lineages_file, 'w', encoding='utf-8') as f:
                    json.dump(lineages[:20], f, indent=2, default=str)  # Top 20 lineages
                if verbose:
                    print(f"[OK] Saved: {lineages_file.name}")
            
            # Save graph data for potential reuse
            graph_file = exp_output / "stn_graph.json"
            if verbose:
                print(f"  Saving graph structure...")
            with open(graph_file, 'w', encoding='utf-8') as f:
                json.dump(builder.to_dict(), f, indent=2, default=str)
            if verbose:
                print(f"[OK] Saved: {graph_file.name}")
                
        except Exception as e:
            if verbose:
                print(f"[WARNING] Error saving lineages/graph: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
        
        main_pbar.update(1)
    
    return all_metrics


def batch_analyze(
    experiments_dir: Path,
    output_dir: Path,
    interactive: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Analyze multiple experiments in a directory.
    
    Args:
        experiments_dir: Directory containing experiment folders
        output_dir: Base output directory
        interactive: Generate interactive visualizations
        verbose: Print progress
        
    Returns:
        Dictionary with results from all experiments
    """
    # Find experiment directories
    exp_dirs = [d for d in experiments_dir.iterdir()
                if d.is_dir() and d.name.startswith('exp-')]
    
    if not exp_dirs:
        print(f"No experiment directories found in {experiments_dir}")
        print("Looking for directories starting with 'exp-'")
        return {}
    
    print(f"Found {len(exp_dirs)} experiment directories")
    
    all_results = {}
    successful = 0
    failed = 0
    
    for exp_dir in sorted(exp_dirs):
        result = analyze_experiment(exp_dir, output_dir, interactive, verbose)
        if result:
            all_results[exp_dir.name] = result
            successful += 1
        else:
            failed += 1
    
    # Create summary report
    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'batch_analysis_timestamp': datetime.now().isoformat(),
            'source_directory': str(experiments_dir),
            'total_experiments': len(exp_dirs),
            'successful': successful,
            'failed': failed,
            'experiments': list(all_results.keys()),
        }, f, indent=2, default=str)
    
    # Create comparison summary
    if all_results:
        comparison = create_comparison_summary(all_results)
        comparison_file = output_dir / "comparison_summary.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\n[OK] Comparison summary saved to: {comparison_file}")
    
    print(f"\n{'='*60}")
    print(f"Batch analysis complete.")
    print(f"  Analyzed: {successful} experiments")
    print(f"  Failed: {failed} experiments")
    print(f"  Results saved to: {output_dir}")
    
    return all_results


def create_comparison_summary(results: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Create a comparison summary across experiments.
    
    Args:
        results: Dictionary of experiment results
        
    Returns:
        Comparison summary dictionary
    """
    comparison = {
        'experiment_count': len(results),
        'best_fitness_per_experiment': {},
        'operator_effectiveness_summary': {},
        'ranking_by_fitness': [],
    }
    
    # Collect per-experiment summaries
    fitness_rankings = []
    for exp_name, metrics in results.items():
        fitness = metrics.get('fitness', {}).get('fitness_max', 0.0)
        comparison['best_fitness_per_experiment'][exp_name] = fitness
        fitness_rankings.append((exp_name, fitness))
        
        # Aggregate operator stats
        op_stats = metrics.get('operator', {}).get('per_operator', {})
        for op, stats in op_stats.items():
            if op not in comparison['operator_effectiveness_summary']:
                comparison['operator_effectiveness_summary'][op] = {
                    'total_uses': 0,
                    'success_rates': [],
                    'improvements': [],
                }
            comparison['operator_effectiveness_summary'][op]['total_uses'] += stats.get('count', 0)
            if stats.get('success_rate', 0) > 0:
                comparison['operator_effectiveness_summary'][op]['success_rates'].append(
                    stats['success_rate']
                )
            if stats.get('mean_improvement', 0) != 0:
                comparison['operator_effectiveness_summary'][op]['improvements'].append(
                    stats['mean_improvement']
                )
    
    # Calculate aggregate operator stats
    import numpy as np
    for op, data in comparison['operator_effectiveness_summary'].items():
        if data['success_rates']:
            data['avg_success_rate'] = float(np.mean(data['success_rates']))
        else:
            data['avg_success_rate'] = 0.0
        if data['improvements']:
            data['avg_improvement'] = float(np.mean(data['improvements']))
        else:
            data['avg_improvement'] = 0.0
        # Clean up lists
        del data['success_rates']
        del data['improvements']
    
    # Ranking by fitness
    fitness_rankings.sort(key=lambda x: x[1], reverse=True)
    comparison['ranking_by_fitness'] = [
        {'rank': i+1, 'experiment': name, 'best_fitness': fit}
        for i, (name, fit) in enumerate(fitness_rankings)
    ]
    
    return comparison


def main():
    """Main entry point for STN Analyzer."""
    parser = argparse.ArgumentParser(
        description="STN Analyzer for MADA-LLAMEA Experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single experiment
  python stn_analyzer.py --exp-dir exp-12-14_100247-google-gemini-2.5-flash-mada-v2

  # Analyze with interactive HTML output
  python stn_analyzer.py --exp-dir path/to/exp --interactive

  # Batch analyze all experiments in parent directory
  python stn_analyzer.py --batch ..

  # Specify custom output directory
  python stn_analyzer.py --exp-dir exp-folder --output-dir my_analysis
        """
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--exp-dir',
        type=Path,
        help='Path to single experiment directory'
    )
    input_group.add_argument(
        '--batch',
        type=Path,
        help='Path to folder containing multiple experiment directories'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('stn_outputs'),
        help='Output directory for visualizations (default: stn_outputs)'
    )
    
    # Visualization options
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Generate interactive HTML visualizations'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if args.exp_dir and not args.exp_dir.exists():
        print(f"Error: Experiment directory not found: {args.exp_dir}")
        sys.exit(1)
    
    if args.batch and not args.batch.exists():
        print(f"Error: Batch directory not found: {args.batch}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run analysis
    verbose = not args.quiet
    
    if verbose:
        print("\n" + "="*60)
        print("STN ANALYZER FOR MADA-LLAMEA")
        print("Search Trajectory Network Analysis Tool")
        print("="*60)
    
    if args.exp_dir:
        result = analyze_experiment(
            args.exp_dir,
            args.output_dir,
            args.interactive,
            verbose
        )
        if result:
            if verbose:
                print(f"\n[OK] Analysis complete!")
                print(f"  Results saved to: {args.output_dir / args.exp_dir.name}")
        else:
            print("\n[ERROR] Analysis failed")
            sys.exit(1)
    
    elif args.batch:
        results = batch_analyze(
            args.batch,
            args.output_dir,
            args.interactive,
            verbose
        )
        if not results:
            print("\n[ERROR] No experiments analyzed successfully")
            sys.exit(1)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Analysis complete!")
        print(f"Output directory: {args.output_dir.absolute()}")
        print("="*60)


if __name__ == "__main__":
    main()

