"""
Simple visualization script for LLaMEA experiment results.
Usage: python visualize_results.py exp-11-27_HHMMSS-model-name/
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from glob import glob

def plot_experiment_results(exp_dir, output_dir=None):
    """
    Visualize results from a single experiment.
    
    Args:
        exp_dir: Path to experiment directory
        output_dir: Output directory for plots (defaults to exp_dir)
    """
    if output_dir is None:
        output_dir = exp_dir
    
    # Find all AUC files
    auc_files = sorted(glob(f"{exp_dir}/try-*-aucs.txt"))
    
    if not auc_files:
        print(f"No AUC files found in {exp_dir}")
        return
    
    print(f"Found {len(auc_files)} algorithm evaluations")
    
    # Extract mean AUC for each algorithm
    mean_aucs = []
    generations = []
    
    for auc_file in auc_files:
        # Extract try number from filename
        try_num = int(os.path.basename(auc_file).split('-')[1])
        generations.append(try_num)
        
        # Load AUC scores (216 values per algorithm)
        aucs = np.loadtxt(auc_file)
        aucs = np.atleast_1d(aucs)  # Handle single values
        if len(aucs) > 0 and aucs[0] != 0:
            mean_aucs.append(np.mean(aucs))
        else:
            mean_aucs.append(0.0)
    
    mean_aucs = np.array(mean_aucs)
    generations = np.array(generations)
    
    # Calculate best-so-far trajectory
    best_so_far = np.maximum.accumulate(mean_aucs)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: All algorithms performance
    axes[0].plot(generations, mean_aucs, 'o-', label='Algorithm Fitness', alpha=0.6)
    axes[0].plot(generations, best_so_far, 'r-', linewidth=2, label='Best So Far')
    axes[0].axhline(y=np.max(mean_aucs), color='g', linestyle='--', alpha=0.3, 
                    label=f'Best: {np.max(mean_aucs):.4f}')
    axes[0].set_xlabel('API Call / Generation', fontsize=12)
    axes[0].set_ylabel('Mean AUC Score', fontsize=12)
    axes[0].set_title('Algorithm Performance Over Time', fontsize=14, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Improvement over baseline
    if len(mean_aucs) > 0:
        baseline = mean_aucs[0]
        improvements = ((mean_aucs - baseline) / (baseline + 1e-10)) * 100
        axes[1].bar(generations, improvements, alpha=0.7, color=['g' if x > 0 else 'r' for x in improvements])
        axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        axes[1].set_xlabel('API Call / Generation', fontsize=12)
        axes[1].set_ylabel('Improvement from First Algorithm (%)', fontsize=12)
        axes[1].set_title('Relative Performance Improvement', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'experiment_results.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    plt.close()
    
    # Print statistics
    print("\n" + "="*60)
    print("EXPERIMENT STATISTICS")
    print("="*60)
    print(f"Total algorithms evaluated: {len(mean_aucs)}")
    print(f"Best fitness: {np.max(mean_aucs):.4f} (API call {np.argmax(mean_aucs)})")
    print(f"First fitness: {mean_aucs[0]:.4f}")
    print(f"Mean fitness: {np.mean(mean_aucs):.4f} ± {np.std(mean_aucs):.4f}")
    print(f"Improvement: {((np.max(mean_aucs) - mean_aucs[0])/mean_aucs[0])*100:.2f}%")
    print("="*60)
    
    # Create detailed box plot by generation
    if len(auc_files) >= 5:
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Sample up to 20 algorithms for box plot
        sample_indices = np.linspace(0, len(auc_files)-1, min(20, len(auc_files)), dtype=int)
        box_data = []
        box_labels = []
        
        for idx in sample_indices:
            aucs = np.loadtxt(auc_files[idx])
            aucs = np.atleast_1d(aucs)  # Handle single values
            if len(aucs) > 0 and aucs[0] != 0:
                box_data.append(aucs)
                box_labels.append(f"API {idx}")
        
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
        
        # Color boxes
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.6)
        
        ax.set_xlabel('API Call', fontsize=12)
        ax.set_ylabel('AUC Score Distribution', fontsize=12)
        ax.set_title('AUC Score Distributions Across BBOB Functions (216 runs each)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        output_file = os.path.join(output_dir, 'auc_distributions.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Distribution plot saved to: {output_file}")
        plt.close()


def plot_comparison(exp_dirs, labels=None, output_file="comparison.png"):
    """
    Compare multiple experiments.
    
    Args:
        exp_dirs: List of experiment directory paths
        labels: List of labels for each experiment
        output_file: Output filename
    """
    if labels is None:
        labels = [os.path.basename(d) for d in exp_dirs]
    
    plt.figure(figsize=(12, 7))
    
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    for i, exp_dir in enumerate(exp_dirs):
        auc_files = sorted(glob(f"{exp_dir}/try-*-aucs.txt"))
        
        if not auc_files:
            print(f"Warning: No AUC files found in {exp_dir}")
            continue
        
        mean_aucs = []
        for auc_file in auc_files:
            aucs = np.loadtxt(auc_file)
            aucs = np.atleast_1d(aucs)  # Handle single values
            if len(aucs) > 0 and aucs[0] != 0:
                mean_aucs.append(np.mean(aucs))
            else:
                mean_aucs.append(0.0)
        
        mean_aucs = np.array(mean_aucs)
        best_so_far = np.maximum.accumulate(mean_aucs)
        
        color = colors[i % len(colors)]
        plt.plot(range(len(best_so_far)), best_so_far, 
                color=color, linewidth=2, label=labels[i], marker='o', markersize=4)
    
    plt.xlabel('API Call / Generation', fontsize=12)
    plt.ylabel('Best AUC Score', fontsize=12)
    plt.title('Comparison of Experiment Runs', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_file}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_results.py <experiment_directory>")
        print("   or: python visualize_results.py <exp_dir1> <exp_dir2> ... (for comparison)")
        sys.exit(1)
    
    exp_dirs = sys.argv[1:]
    
    if len(exp_dirs) == 1:
        # Single experiment visualization
        plot_experiment_results(exp_dirs[0])
    else:
        # Multiple experiments comparison
        print(f"Comparing {len(exp_dirs)} experiments...")
        plot_comparison(exp_dirs)
        print("\nIndividual experiment results:")
        for exp_dir in exp_dirs:
            print(f"\n{exp_dir}:")
            plot_experiment_results(exp_dir)

