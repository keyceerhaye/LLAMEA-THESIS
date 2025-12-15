#!/usr/bin/env python3
"""
Quick analyzer script with enhanced error reporting and debugging.

This script provides a simple way to analyze a single experiment with
detailed progress logging and error handling.
"""

import sys
import argparse
from pathlib import Path
import time
import traceback

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from stn.data_loader import ExperimentDataLoader
from stn.graph_builder import STNGraphBuilder
from stn.metrics import STNMetrics
from stn.visualizer import STNVisualizer


def analyze_with_debugging(exp_dir: Path, output_dir: Path, interactive: bool = False):
    """Analyze experiment with detailed debugging output."""
    
    print("="*80)
    print("STN ANALYZER - DEBUG MODE")
    print("="*80)
    print(f"\nExperiment directory: {exp_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Interactive: {interactive}")
    print()
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # ====================
        # STEP 1: Load Data
        # ====================
        print("\n" + "="*80)
        print("STEP 1: LOADING DATA")
        print("="*80)
        step1_start = time.time()
        
        print(f"\nInitializing data loader...")
        loader = ExperimentDataLoader(exp_dir)
        
        print(f"Loading experiment data...")
        nodes, edges = loader.load()
        
        step1_duration = time.time() - step1_start
        print(f"\n[OK] STEP 1 COMPLETE (took {step1_duration:.1f}s)")
        print(f"  Loaded {len(nodes)} nodes and {len(edges)} edges")
        
        if not nodes:
            print("\n[ERROR] No data loaded!")
            return False
        
        # ====================
        # STEP 2: Build Graph
        # ====================
        print("\n" + "="*80)
        print("STEP 2: BUILDING GRAPH")
        print("="*80)
        step2_start = time.time()
        
        print(f"\nBuilding graph from {len(nodes)} nodes and {len(edges)} edges...")
        builder = STNGraphBuilder()
        graph = builder.build_graph(nodes, edges)
        
        step2_duration = time.time() - step2_start
        print(f"\n[OK] STEP 2 COMPLETE (took {step2_duration:.1f}s)")
        print(f"  Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        
        # ====================
        # STEP 3: Calculate Metrics
        # ====================
        print("\n" + "="*80)
        print("STEP 3: CALCULATING METRICS")
        print("="*80)
        step3_start = time.time()
        
        print(f"\nInitializing metrics calculator...")
        metrics = STNMetrics(graph)
        
        print(f"\n  Computing basic metrics...")
        basic = metrics.basic_metrics()
        print(f"    [OK] Basic metrics computed")
        
        print(f"\n  Computing operator analysis...")
        operator_stats = metrics.operator_analysis()
        print(f"    [OK] Operator analysis computed")
        
        print(f"\n  Computing trajectory analysis...")
        trajectory_stats = metrics.trajectory_analysis()
        print(f"    [OK] Trajectory analysis computed")
        
        print(f"\n  Computing fitness landscape...")
        fitness_stats = metrics.fitness_landscape_analysis()
        print(f"    [OK] Fitness landscape computed")
        
        print(f"\n  Computing diversity analysis...")
        diversity_stats = metrics.diversity_analysis()
        print(f"    [OK] Diversity analysis computed")
        
        print(f"\n  Computing bandit analysis...")
        bandit_stats = metrics.bandit_analysis()
        print(f"    [OK] Bandit analysis computed")
        
        print(f"\n  Identifying successful lineages...")
        lineages = metrics.identify_successful_lineages()
        print(f"    [OK] Found {len(lineages)} lineages")
        
        step3_duration = time.time() - step3_start
        print(f"\n[OK] STEP 3 COMPLETE (took {step3_duration:.1f}s)")
        
        # Print summary
        print(f"\n--- Metrics Summary ---")
        print(f"  Nodes: {basic['num_nodes']}, Edges: {basic['num_edges']}")
        print(f"  Best Fitness: {fitness_stats.get('fitness_max', 0):.4f}")
        print(f"  Generations: {fitness_stats.get('generation_stats', {}) and max(fitness_stats['generation_stats'].keys(), default=0) + 1}")
        
        # ====================
        # STEP 4: Create Visualizations
        # ====================
        print("\n" + "="*80)
        print("STEP 4: CREATING VISUALIZATIONS")
        print("="*80)
        step4_start = time.time()
        
        exp_output = output_dir / exp_dir.name
        exp_output.mkdir(parents=True, exist_ok=True)
        
        print(f"\nOutput directory: {exp_output}")
        print(f"Initializing visualizer...")
        viz = STNVisualizer(graph, exp_output)
        
        print(f"\nGenerating visualizations...")
        viz_outputs = viz.generate_all_visualizations(
            lineages=lineages,
            interactive=interactive,
        )
        
        step4_duration = time.time() - step4_start
        successful_viz = len([v for v in viz_outputs.values() if v])
        print(f"\n[OK] STEP 4 COMPLETE (took {step4_duration:.1f}s)")
        print(f"  Generated {successful_viz}/{len(viz_outputs)} visualizations")
        
        # ====================
        # STEP 5: Save Results
        # ====================
        print("\n" + "="*80)
        print("STEP 5: SAVING RESULTS")
        print("="*80)
        step5_start = time.time()
        
        import json
        from datetime import datetime
        
        # Save metrics
        print(f"\nSaving metrics to JSON...")
        metrics_file = exp_output / "stn_metrics.json"
        all_metrics = {
            'experiment_name': exp_dir.name,
            'analysis_timestamp': datetime.now().isoformat(),
            'summary': loader.get_experiment_summary(),
            'basic': basic,
            'operator': operator_stats,
            'trajectory': trajectory_stats,
            'fitness': {k: v for k, v in fitness_stats.items() if k != 'generation_stats'},
            'diversity': diversity_stats,
            'bandit': bandit_stats,
            'num_successful_lineages': len(lineages),
        }
        
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        print(f"  [OK] Saved: {metrics_file}")
        
        # Save generation stats separately
        gen_stats_file = exp_output / "generation_stats.json"
        with open(gen_stats_file, 'w', encoding='utf-8') as f:
            json.dump(fitness_stats.get('generation_stats', {}), f, indent=2, default=str)
        print(f"  [OK] Saved: {gen_stats_file}")
        
        # Save lineages
        if lineages:
            lineages_file = exp_output / "successful_lineages.json"
            with open(lineages_file, 'w', encoding='utf-8') as f:
                json.dump(lineages[:20], f, indent=2, default=str)
            print(f"  [OK] Saved: {lineages_file}")
        
        # Save graph
        graph_file = exp_output / "stn_graph.json"
        with open(graph_file, 'w', encoding='utf-8') as f:
            json.dump(builder.to_dict(), f, indent=2, default=str)
        print(f"  [OK] Saved: {graph_file}")
        
        step5_duration = time.time() - step5_start
        print(f"\n[OK] STEP 5 COMPLETE (took {step5_duration:.1f}s)")
        
        # ====================
        # Final Summary
        # ====================
        total_duration = step1_duration + step2_duration + step3_duration + step4_duration + step5_duration
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nTotal time: {total_duration:.1f}s")
        print(f"  Step 1 (Load data):      {step1_duration:.1f}s")
        print(f"  Step 2 (Build graph):    {step2_duration:.1f}s")
        print(f"  Step 3 (Metrics):        {step3_duration:.1f}s")
        print(f"  Step 4 (Visualizations): {step4_duration:.1f}s")
        print(f"  Step 5 (Save results):   {step5_duration:.1f}s")
        print(f"\nResults saved to: {exp_output}")
        print("="*80)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] BY USER")
        print("The analysis was stopped. Check the output above to see where it stopped.")
        return False
        
    except Exception as e:
        print(f"\n\n[FATAL ERROR] {type(e).__name__}: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\n" + "="*80)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Analyze MADA-LLAMEA experiment with detailed debugging output"
    )
    parser.add_argument(
        'exp_dir',
        type=Path,
        help='Path to experiment directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('stn_outputs'),
        help='Output directory (default: stn_outputs)'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Generate interactive HTML visualization'
    )
    
    args = parser.parse_args()
    
    # Validate experiment directory
    if not args.exp_dir.exists():
        print(f"Error: Experiment directory not found: {args.exp_dir}")
        sys.exit(1)
    
    if not args.exp_dir.is_dir():
        print(f"Error: Not a directory: {args.exp_dir}")
        sys.exit(1)
    
    # Run analysis
    success = analyze_with_debugging(args.exp_dir, args.output_dir, args.interactive)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
