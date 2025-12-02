"""
Original LLaMEA-style visualization script.
Adapted from misc/plot_aucs.py for use with thesis experiments.

Usage:
    python plot_llamea_style.py exp-dir1 exp-dir2 ...
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import sys
from glob import glob

def plot_llamea_comparison(exp_dirs, labels=None, colors=None, linestyles=None, 
                           budget=100, output_file="llamea_comparison.png"):
    """
    Plot comparison in original LLaMEA paper style.
    
    Args:
        exp_dirs: List of lists of experiment directories (for replication runs)
                 Can also be list of single directories
        labels: Labels for each experiment
        colors: Colors for each experiment
        linestyles: Line styles for each experiment
        budget: Number of generations to plot
        output_file: Output filename
    """
    # Convert single directories to list of lists
    if isinstance(exp_dirs[0], str):
        exp_dirs = [[d] for d in exp_dirs]
    
    # Default styling
    if colors is None:
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    if linestyles is None:
        linestyles = ['solid'] * len(exp_dirs)
    if labels is None:
        labels = [os.path.basename(dirs[0]) for dirs in exp_dirs]
    
    fig = plt.figure(figsize=(10, 6))
    
    for i in range(len(exp_dirs)):
        exp_dir = exp_dirs[i]
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        label = labels[i]
        
        mean_aucs = []
        std_aucs = []
        
        for k in range(budget):
            m_aucs = []
            for d in exp_dir:
                auc_file = f"{d}/try-{k}-aucs.txt"
                if os.path.isfile(auc_file):
                    aucs = np.loadtxt(auc_file)
                    aucs = np.atleast_1d(aucs)
                    if len(aucs) > 0 and aucs[0] != 0:
                        m_aucs.append(np.mean(aucs))
            
            if len(m_aucs) > 0:
                mean_aucs.append(np.mean(m_aucs))
                std_aucs.append(np.std(m_aucs))
            else:
                mean_aucs.append(np.nan)
                std_aucs.append(np.nan)
        
        mean_aucs = np.array(mean_aucs)
        std_aucs = np.array(std_aucs)
        x = np.arange(len(mean_aucs))
        
        # Plot mean line with shaded std dev
        plt.plot(x, mean_aucs, color=color, linestyle=ls, label=label, linewidth=2)
        plt.fill_between(x, mean_aucs - std_aucs, mean_aucs + std_aucs, 
                        color=color, alpha=0.15)
    
    plt.xlabel('Generation', fontsize=13)
    plt.ylabel('AOCC Score', fontsize=13)
    plt.title('LLaMEA Performance Comparison', fontsize=15, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 0.7)  # Standard AOCC range
    plt.xlim(0, budget)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ LLaMEA-style plot saved to: {output_file}")
    plt.close()


def plot_best_trajectory(exp_dirs, labels=None, colors=None, linestyles=None,
                         budget=100, output_file="llamea_best_trajectory.png"):
    """
    Plot best-so-far trajectory (cumulative best).
    
    This is the standard plot used in LLaMEA papers.
    """
    # Convert single directories to list of lists
    if isinstance(exp_dirs[0], str):
        exp_dirs = [[d] for d in exp_dirs]
    
    # Default styling
    if colors is None:
        colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    if linestyles is None:
        linestyles = ['solid'] * len(exp_dirs)
    if labels is None:
        labels = [os.path.basename(dirs[0]) for dirs in exp_dirs]
    
    fig = plt.figure(figsize=(10, 6))
    
    for i in range(len(exp_dirs)):
        exp_dir = exp_dirs[i]
        color = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        label = labels[i]
        
        mean_aucs = []
        std_aucs = []
        
        for k in range(budget):
            m_aucs = []
            for j in range(len(exp_dir)):
                d = exp_dir[j]
                current_best = 0
                
                # Track cumulative best
                for gen in range(k + 1):
                    auc_file = f"{d}/try-{gen}-aucs.txt"
                    if os.path.isfile(auc_file):
                        aucs = np.loadtxt(auc_file)
                        aucs = np.atleast_1d(aucs)
                        if len(aucs) > 0 and aucs[0] != 0:
                            current_best = max(current_best, np.mean(aucs))
                
                m_aucs.append(current_best)
            
            if len(m_aucs) > 0:
                mean_aucs.append(np.mean(m_aucs))
                std_aucs.append(np.std(m_aucs))
            else:
                mean_aucs.append(np.nan)
                std_aucs.append(np.nan)
        
        mean_aucs = np.array(mean_aucs)
        std_aucs = np.array(std_aucs)
        x = np.arange(len(mean_aucs))
        
        # Plot mean line with shaded std dev
        plt.plot(x, mean_aucs, color=color, linestyle=ls, label=label, linewidth=2)
        plt.fill_between(x, mean_aucs - std_aucs, mean_aucs + std_aucs,
                        color=color, alpha=0.15)
    
    plt.xlabel('Generation', fontsize=13)
    plt.ylabel('Best AOCC Score', fontsize=13)
    plt.title('LLaMEA Best-So-Far Performance', fontsize=15, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0.0, 0.7)
    plt.xlim(0, budget)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Best trajectory plot saved to: {output_file}")
    plt.close()


def analyze_experiment(exp_dir):
    """
    Print statistics for a single experiment (original LLaMEA style).
    """
    auc_files = sorted(glob(f"{exp_dir}/try-*-aucs.txt"))
    
    if not auc_files:
        print(f"No AUC files found in {exp_dir}")
        return
    
    aucs_all = []
    best_fitness = 0
    best_gen = 0
    
    for i, auc_file in enumerate(auc_files):
        aucs = np.loadtxt(auc_file)
        aucs = np.atleast_1d(aucs)
        if len(aucs) > 0 and aucs[0] != 0:
            mean_auc = np.mean(aucs)
            aucs_all.append(mean_auc)
            if mean_auc > best_fitness:
                best_fitness = mean_auc
                best_gen = i
        else:
            aucs_all.append(0.0)
    
    print(f"\n{'='*70}")
    print(f"Experiment: {os.path.basename(exp_dir)}")
    print(f"{'='*70}")
    print(f"Generations evaluated: {len(aucs_all)}")
    print(f"Best AOCC: {best_fitness:.4f} (Generation {best_gen})")
    print(f"Mean AOCC: {np.mean(aucs_all):.4f} ± {np.std(aucs_all):.4f}")
    print(f"Final AOCC: {aucs_all[-1]:.4f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single experiment analysis:")
        print("    python plot_llamea_style.py exp-dir/")
        print()
        print("  Compare multiple experiments:")
        print("    python plot_llamea_style.py exp-dir1/ exp-dir2/ exp-dir3/")
        print()
        print("  With custom labels:")
        print("    python plot_llamea_style.py exp-dir1/ exp-dir2/ --labels 'Gemini' 'GPT-4'")
        sys.exit(1)
    
    # Parse arguments
    exp_dirs = []
    labels = None
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--labels':
            labels = []
            i += 1
            while i < len(sys.argv) and not sys.argv[i].startswith('--'):
                labels.append(sys.argv[i])
                i += 1
        elif not sys.argv[i].startswith('--'):
            exp_dirs.append(sys.argv[i])
            i += 1
        else:
            i += 1
    
    if not exp_dirs:
        print("Error: No experiment directories specified")
        sys.exit(1)
    
    # Determine budget from first experiment
    budget = 100
    first_exp = exp_dirs[0]
    auc_files = glob(f"{first_exp}/try-*-aucs.txt")
    if auc_files:
        budget = min(100, len(auc_files))
    
    print(f"Found {len(exp_dirs)} experiment(s)")
    print(f"Budget: {budget} generations")
    print()
    
    # Analyze each experiment
    for exp_dir in exp_dirs:
        analyze_experiment(exp_dir)
    
    # Create plots
    if len(exp_dirs) == 1:
        print("Single experiment - creating individual plots...")
        
        # Use basename as label
        if labels is None:
            labels = [os.path.basename(exp_dirs[0])]
        
        # Mean trajectory
        plot_llamea_comparison([exp_dirs[0]], labels=labels, budget=budget,
                              output_file=f"{exp_dirs[0]}/llamea_trajectory.png")
        
        # Best-so-far trajectory
        plot_best_trajectory([exp_dirs[0]], labels=labels, budget=budget,
                            output_file=f"{exp_dirs[0]}/llamea_best_trajectory.png")
    else:
        print("Multiple experiments - creating comparison plots...")
        
        # Mean trajectory comparison
        plot_llamea_comparison(exp_dirs, labels=labels, budget=budget,
                              output_file="llamea_comparison.png")
        
        # Best-so-far trajectory comparison
        plot_best_trajectory(exp_dirs, labels=labels, budget=budget,
                            output_file="llamea_best_trajectory.png")
    
    print("\n✅ Done!")

