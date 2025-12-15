"""
Visualize algorithm performance on multimodal BBOB functions only.

BBOB Function Groups:
- f1-f5: Separable (lines 1-45)
- f6-f9: Low/moderate conditioning (lines 46-81)
- f10-f14: High conditioning, unimodal (lines 82-126)
- f15-f19: Multi-modal with adequate global structure (lines 127-171)
- f20-f24: Multi-modal with weak global structure (lines 172-216)

Multimodal functions: f15-f24 (lines 127-216)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path


# Line indices for each function group (0-indexed)
FUNCTION_GROUPS = {
    'separable': (0, 45),           # f1-f5: 5 funcs × 3 instances × 3 reps = 45
    'low_conditioning': (45, 81),   # f6-f9: 4 funcs × 3 instances × 3 reps = 36
    'high_conditioning': (81, 126), # f10-f14: 5 funcs × 3 instances × 3 reps = 45
    'multimodal_adequate': (126, 171),  # f15-f19: 5 funcs × 3 instances × 3 reps = 45
    'multimodal_weak': (171, 216),      # f20-f24: 5 funcs × 3 instances × 3 reps = 45
}

# Combined multimodal (f15-f24)
MULTIMODAL_RANGE = (126, 216)  # f15-f24


def load_aucs(exp_dir, budget=100):
    """Load all AUC files from an experiment directory."""
    all_aucs = []
    multimodal_aucs = []
    multimodal_adequate_aucs = []
    multimodal_weak_aucs = []
    
    for k in range(budget):
        auc_file = os.path.join(exp_dir, f"try-{k}-aucs.txt")
        if os.path.isfile(auc_file):
            try:
                aucs = np.loadtxt(auc_file)
                # Handle 0-dimensional arrays (single value)
                if aucs.ndim == 0:
                    aucs = np.array([float(aucs)])
                
                # Check if we have the full 216 values (24 funcs × 3 instances × 3 reps)
                if len(aucs) >= 216:
                    all_aucs.append(np.mean(aucs))
                    
                    # Extract multimodal (f15-f24)
                    mm_aucs = aucs[MULTIMODAL_RANGE[0]:MULTIMODAL_RANGE[1]]
                    multimodal_aucs.append(np.mean(mm_aucs))
                    
                    # Extract multimodal adequate (f15-f19)
                    mm_adq = aucs[FUNCTION_GROUPS['multimodal_adequate'][0]:FUNCTION_GROUPS['multimodal_adequate'][1]]
                    multimodal_adequate_aucs.append(np.mean(mm_adq))
                    
                    # Extract multimodal weak (f20-f24)
                    mm_weak = aucs[FUNCTION_GROUPS['multimodal_weak'][0]:FUNCTION_GROUPS['multimodal_weak'][1]]
                    multimodal_weak_aucs.append(np.mean(mm_weak))
                else:
                    # Partial or error file - just use overall mean
                    all_aucs.append(np.mean(aucs) if len(aucs) > 0 else np.nan)
                    multimodal_aucs.append(np.nan)
                    multimodal_adequate_aucs.append(np.nan)
                    multimodal_weak_aucs.append(np.nan)
            except Exception as e:
                all_aucs.append(np.nan)
                multimodal_aucs.append(np.nan)
                multimodal_adequate_aucs.append(np.nan)
                multimodal_weak_aucs.append(np.nan)
        else:
            all_aucs.append(np.nan)
            multimodal_aucs.append(np.nan)
            multimodal_adequate_aucs.append(np.nan)
            multimodal_weak_aucs.append(np.nan)
    
    return {
        'all': np.array(all_aucs),
        'multimodal': np.array(multimodal_aucs),
        'multimodal_adequate': np.array(multimodal_adequate_aucs),
        'multimodal_weak': np.array(multimodal_weak_aucs),
    }


def load_per_function_aucs(exp_dir, budget=100):
    """Load AUCs per function group for detailed analysis."""
    results = {group: [] for group in FUNCTION_GROUPS.keys()}
    
    for k in range(budget):
        auc_file = os.path.join(exp_dir, f"try-{k}-aucs.txt")
        if os.path.isfile(auc_file):
            try:
                aucs = np.loadtxt(auc_file)
                # Handle 0-dimensional arrays (single value)
                if aucs.ndim == 0:
                    aucs = np.array([float(aucs)])
                
                if len(aucs) >= 216:
                    for group, (start, end) in FUNCTION_GROUPS.items():
                        group_aucs = aucs[start:end]
                        results[group].append(np.mean(group_aucs))
                else:
                    for group in FUNCTION_GROUPS.keys():
                        results[group].append(np.nan)
            except Exception:
                for group in FUNCTION_GROUPS.keys():
                    results[group].append(np.nan)
        else:
            for group in FUNCTION_GROUPS.keys():
                results[group].append(np.nan)
    
    return {k: np.array(v) for k, v in results.items()}


def get_best_algorithm_multimodal(exp_dir, budget=100):
    """Find the best algorithm for multimodal functions."""
    best_idx = -1
    best_score = -1
    
    for k in range(budget):
        auc_file = os.path.join(exp_dir, f"try-{k}-aucs.txt")
        if os.path.isfile(auc_file):
            try:
                aucs = np.loadtxt(auc_file)
                # Handle 0-dimensional arrays (single value)
                if aucs.ndim == 0:
                    aucs = np.array([float(aucs)])
                
                if len(aucs) >= 216:
                    mm_aucs = aucs[MULTIMODAL_RANGE[0]:MULTIMODAL_RANGE[1]]
                    mm_score = np.mean(mm_aucs)
                    
                    if mm_score > best_score:
                        best_score = mm_score
                        best_idx = k
            except Exception:
                pass
    
    return best_idx, best_score


def plot_convergence(exp_dir, output_dir=None, budget=100):
    """Plot convergence curves for all vs multimodal functions."""
    data = load_aucs(exp_dir, budget)
    
    if output_dir is None:
        output_dir = exp_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate cumulative best
    all_best = np.maximum.accumulate(np.nan_to_num(data['all'], nan=0))
    mm_best = np.maximum.accumulate(np.nan_to_num(data['multimodal'], nan=0))
    mm_adq_best = np.maximum.accumulate(np.nan_to_num(data['multimodal_adequate'], nan=0))
    mm_weak_best = np.maximum.accumulate(np.nan_to_num(data['multimodal_weak'], nan=0))
    
    x = np.arange(len(data['all']))
    
    # Plot 1: All vs Multimodal convergence
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, all_best, 'b-', label='All Functions (f1-f24)', linewidth=2)
    ax.plot(x, mm_best, 'r-', label='Multimodal Only (f15-f24)', linewidth=2)
    ax.set_xlabel('API Calls', fontsize=12)
    ax.set_ylabel('Best AOCC', fontsize=12)
    ax.set_title('Convergence: All Functions vs Multimodal Functions', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_multimodal.png'), dpi=150)
    plt.close()
    
    # Plot 2: Multimodal subgroups
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, mm_adq_best, 'g-', label='f15-f19: Multimodal (adequate structure)', linewidth=2)
    ax.plot(x, mm_weak_best, 'm-', label='f20-f24: Multimodal (weak structure)', linewidth=2)
    ax.plot(x, mm_best, 'r--', label='f15-f24: All Multimodal', linewidth=2, alpha=0.7)
    ax.set_xlabel('API Calls', fontsize=12)
    ax.set_ylabel('Best AOCC', fontsize=12)
    ax.set_title('Convergence: Multimodal Function Subgroups', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_multimodal_subgroups.png'), dpi=150)
    plt.close()
    
    # Plot 3: Per-iteration performance (not cumulative best)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(x, data['multimodal'], c='red', alpha=0.5, s=20, label='Multimodal (f15-f24)')
    ax.plot(x, mm_best, 'r-', linewidth=2, label='Best So Far')
    ax.set_xlabel('API Calls', fontsize=12)
    ax.set_ylabel('AOCC', fontsize=12)
    ax.set_title('Per-Algorithm Multimodal Performance', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_multimodal.png'), dpi=150)
    plt.close()
    
    print(f"Plots saved to {output_dir}")


def plot_function_group_comparison(exp_dir, output_dir=None, budget=100):
    """Plot comparison across all function groups."""
    data = load_per_function_aucs(exp_dir, budget)
    
    if output_dir is None:
        output_dir = exp_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    x = np.arange(budget)
    
    # Calculate cumulative best for each group
    group_labels = {
        'separable': 'f1-f5: Separable',
        'low_conditioning': 'f6-f9: Low Conditioning',
        'high_conditioning': 'f10-f14: High Conditioning',
        'multimodal_adequate': 'f15-f19: Multimodal (adequate)',
        'multimodal_weak': 'f20-f24: Multimodal (weak)',
    }
    
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for (group, label), color in zip(group_labels.items(), colors):
        group_data = data[group]
        best_so_far = np.maximum.accumulate(np.nan_to_num(group_data, nan=0))
        ax.plot(x[:len(best_so_far)], best_so_far, '-', color=color, label=label, linewidth=2)
    
    ax.set_xlabel('API Calls', fontsize=12)
    ax.set_ylabel('Best AOCC', fontsize=12)
    ax.set_title('Convergence by BBOB Function Group', fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_all_groups.png'), dpi=150)
    plt.close()
    
    # Bar chart: Final best per group
    fig, ax = plt.subplots(figsize=(10, 6))
    
    final_scores = []
    labels = []
    for group, label in group_labels.items():
        group_data = data[group]
        best = np.nanmax(group_data) if len(group_data) > 0 else 0
        final_scores.append(best)
        labels.append(label)
    
    bars = ax.bar(labels, final_scores, color=colors, edgecolor='black')
    ax.set_ylabel('Best AOCC', fontsize=12)
    ax.set_title('Best Performance by Function Group', fontsize=14)
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, score in zip(bars, final_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bar_chart_groups.png'), dpi=150)
    plt.close()
    
    print(f"Function group plots saved to {output_dir}")


def print_summary(exp_dir, budget=100):
    """Print summary statistics for multimodal functions."""
    data = load_aucs(exp_dir, budget)
    group_data = load_per_function_aucs(exp_dir, budget)
    
    print("\n" + "="*60)
    print("MULTIMODAL FUNCTION ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nExperiment: {os.path.basename(exp_dir)}")
    print(f"Valid evaluations: {np.sum(~np.isnan(data['all']))}")
    
    print("\n--- Best AOCC by Function Group ---")
    print(f"  All Functions (f1-f24):        {np.nanmax(data['all']):.4f}")
    print(f"  Multimodal All (f15-f24):      {np.nanmax(data['multimodal']):.4f}")
    print(f"  Multimodal Adequate (f15-f19): {np.nanmax(data['multimodal_adequate']):.4f}")
    print(f"  Multimodal Weak (f20-f24):     {np.nanmax(data['multimodal_weak']):.4f}")
    
    print("\n--- All Function Groups ---")
    for group, (start, end) in FUNCTION_GROUPS.items():
        best = np.nanmax(group_data[group])
        mean = np.nanmean(group_data[group])
        print(f"  {group:25s}: best={best:.4f}, mean={mean:.4f}")
    
    best_idx, best_score = get_best_algorithm_multimodal(exp_dir, budget)
    print(f"\n--- Best Algorithm for Multimodal ---")
    print(f"  Algorithm index: try-{best_idx}")
    print(f"  Multimodal AOCC: {best_score:.4f}")
    
    # Check if code file exists
    code_dir = os.path.join(exp_dir, "code")
    if os.path.isdir(code_dir):
        for f in os.listdir(code_dir):
            if f.startswith(f"try-{best_idx}-"):
                print(f"  Code file: {f}")
                break
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize algorithm performance on multimodal BBOB functions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python visualize_multimodal.py path/to/experiment
    python visualize_multimodal.py path/to/experiment --output plots/
    python visualize_multimodal.py path/to/experiment --budget 50 --summary-only
        """
    )
    
    parser.add_argument('exp_dir', type=str, help='Experiment directory path')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output directory for plots (default: same as exp_dir)')
    parser.add_argument('--budget', '-b', type=int, default=100,
                       help='Maximum number of algorithm evaluations to analyze')
    parser.add_argument('--summary-only', '-s', action='store_true',
                       help='Only print summary, no plots')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.exp_dir):
        print(f"Error: Directory not found: {args.exp_dir}")
        return
    
    print_summary(args.exp_dir, args.budget)
    
    if not args.summary_only:
        plot_convergence(args.exp_dir, args.output, args.budget)
        plot_function_group_comparison(args.exp_dir, args.output, args.budget)


if __name__ == "__main__":
    main()
