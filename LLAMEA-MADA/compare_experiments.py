"""
Compare multiple experiment results on BBOB functions.

Usage:
    python compare_experiments.py exp1 exp2 exp3 --labels "Baseline" "GA-LLAMEA" "MADA"
    python compare_experiments.py exp1 exp2 --output comparison_plots/
    python compare_experiments.py exp1 exp2 --multimodal-only
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
from typing import List, Dict, Optional


# Line indices for each function group (0-indexed)
FUNCTION_GROUPS = {
    'separable': (0, 45),           # f1-f5: 5 funcs × 3 instances × 3 reps = 45
    'low_conditioning': (45, 81),   # f6-f9: 4 funcs × 3 instances × 3 reps = 36
    'high_conditioning': (81, 126), # f10-f14: 5 funcs × 3 instances × 3 reps = 45
    'multimodal_adequate': (126, 171),  # f15-f19: 5 funcs × 3 instances × 3 reps = 45
    'multimodal_weak': (171, 216),      # f20-f24: 5 funcs × 3 instances × 3 reps = 45
}

GROUP_DISPLAY_NAMES = {
    'separable': 'f1-f5: Separable',
    'low_conditioning': 'f6-f9: Low Conditioning',
    'high_conditioning': 'f10-f14: High Conditioning',
    'multimodal_adequate': 'f15-f19: Multimodal (adequate)',
    'multimodal_weak': 'f20-f24: Multimodal (weak)',
}

# Combined multimodal (f15-f24)
MULTIMODAL_RANGE = (126, 216)

# Color palette for experiments (colorblind-friendly)
EXPERIMENT_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]

LINE_STYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
MARKERS = ['o', 's', '^', 'D', 'v', '<', '>', 'p', 'h', '*']


def load_experiment_data(exp_dir: str, budget: int = 100) -> Dict:
    """Load all AUC data from an experiment directory."""
    data = {
        'all': [],
        'multimodal': [],
        'multimodal_adequate': [],
        'multimodal_weak': [],
        'groups': {group: [] for group in FUNCTION_GROUPS.keys()},
    }
    
    for k in range(budget):
        auc_file = os.path.join(exp_dir, f"try-{k}-aucs.txt")
        if os.path.isfile(auc_file):
            try:
                aucs = np.loadtxt(auc_file)
                if aucs.ndim == 0:
                    aucs = np.array([float(aucs)])
                
                if len(aucs) >= 216:
                    data['all'].append(np.mean(aucs))
                    
                    # Multimodal ranges
                    mm_aucs = aucs[MULTIMODAL_RANGE[0]:MULTIMODAL_RANGE[1]]
                    data['multimodal'].append(np.mean(mm_aucs))
                    
                    mm_adq = aucs[FUNCTION_GROUPS['multimodal_adequate'][0]:FUNCTION_GROUPS['multimodal_adequate'][1]]
                    data['multimodal_adequate'].append(np.mean(mm_adq))
                    
                    mm_weak = aucs[FUNCTION_GROUPS['multimodal_weak'][0]:FUNCTION_GROUPS['multimodal_weak'][1]]
                    data['multimodal_weak'].append(np.mean(mm_weak))
                    
                    # Per-group
                    for group, (start, end) in FUNCTION_GROUPS.items():
                        data['groups'][group].append(np.mean(aucs[start:end]))
                else:
                    data['all'].append(np.mean(aucs) if len(aucs) > 0 else np.nan)
                    data['multimodal'].append(np.nan)
                    data['multimodal_adequate'].append(np.nan)
                    data['multimodal_weak'].append(np.nan)
                    for group in FUNCTION_GROUPS.keys():
                        data['groups'][group].append(np.nan)
            except Exception:
                data['all'].append(np.nan)
                data['multimodal'].append(np.nan)
                data['multimodal_adequate'].append(np.nan)
                data['multimodal_weak'].append(np.nan)
                for group in FUNCTION_GROUPS.keys():
                    data['groups'][group].append(np.nan)
        else:
            data['all'].append(np.nan)
            data['multimodal'].append(np.nan)
            data['multimodal_adequate'].append(np.nan)
            data['multimodal_weak'].append(np.nan)
            for group in FUNCTION_GROUPS.keys():
                data['groups'][group].append(np.nan)
    
    # Convert to numpy arrays
    data['all'] = np.array(data['all'])
    data['multimodal'] = np.array(data['multimodal'])
    data['multimodal_adequate'] = np.array(data['multimodal_adequate'])
    data['multimodal_weak'] = np.array(data['multimodal_weak'])
    for group in FUNCTION_GROUPS.keys():
        data['groups'][group] = np.array(data['groups'][group])
    
    return data


def get_best_algorithm(exp_dir: str, budget: int = 100, metric: str = 'all') -> tuple:
    """Find the best algorithm for a given metric."""
    best_idx = -1
    best_score = -1
    
    for k in range(budget):
        auc_file = os.path.join(exp_dir, f"try-{k}-aucs.txt")
        if os.path.isfile(auc_file):
            try:
                aucs = np.loadtxt(auc_file)
                if aucs.ndim == 0:
                    aucs = np.array([float(aucs)])
                
                if len(aucs) >= 216:
                    if metric == 'all':
                        score = np.mean(aucs)
                    elif metric == 'multimodal':
                        score = np.mean(aucs[MULTIMODAL_RANGE[0]:MULTIMODAL_RANGE[1]])
                    elif metric in FUNCTION_GROUPS:
                        start, end = FUNCTION_GROUPS[metric]
                        score = np.mean(aucs[start:end])
                    else:
                        score = np.mean(aucs)
                    
                    if score > best_score:
                        best_score = score
                        best_idx = k
            except Exception:
                pass
    
    # Get algorithm name
    alg_name = f"try-{best_idx}"
    code_dir = os.path.join(exp_dir, "code")
    if os.path.isdir(code_dir):
        for f in os.listdir(code_dir):
            if f.startswith(f"try-{best_idx}-"):
                alg_name = f.replace(".py", "")
                break
    
    return best_idx, best_score, alg_name


def plot_convergence_comparison(
    experiments: List[Dict],
    output_dir: str,
    metric: str = 'all',
    title: str = None
):
    """Plot convergence comparison for multiple experiments."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    for i, exp in enumerate(experiments):
        data = exp['data']
        label = exp['label']
        color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
        linestyle = LINE_STYLES[i % len(LINE_STYLES)]
        
        if metric == 'all':
            values = data['all']
        elif metric == 'multimodal':
            values = data['multimodal']
        elif metric == 'multimodal_adequate':
            values = data['multimodal_adequate']
        elif metric == 'multimodal_weak':
            values = data['multimodal_weak']
        else:
            values = data['groups'].get(metric, data['all'])
        
        # Calculate cumulative best
        best_so_far = np.maximum.accumulate(np.nan_to_num(values, nan=0))
        x = np.arange(len(best_so_far))
        
        ax.plot(x, best_so_far, color=color, linestyle=linestyle, 
                label=label, linewidth=2.5)
    
    metric_titles = {
        'all': 'All Functions (f1-f24)',
        'multimodal': 'Multimodal Functions (f15-f24)',
        'multimodal_adequate': 'Multimodal Adequate (f15-f19)',
        'multimodal_weak': 'Multimodal Weak (f20-f24)',
    }
    
    ax.set_xlabel('API Calls', fontsize=13)
    ax.set_ylabel('Best AOCC', fontsize=13)
    ax.set_title(title or f'Convergence Comparison: {metric_titles.get(metric, metric)}', fontsize=14)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(len(exp['data']['all']) for exp in experiments))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{metric}.png'), dpi=150)
    plt.close()


def plot_all_groups_comparison(experiments: List[Dict], output_dir: str):
    """Plot convergence for all function groups in a grid."""
    groups = list(FUNCTION_GROUPS.keys())
    n_groups = len(groups)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, group in enumerate(groups):
        ax = axes[idx]
        
        for i, exp in enumerate(experiments):
            data = exp['data']
            label = exp['label']
            color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
            linestyle = LINE_STYLES[i % len(LINE_STYLES)]
            
            values = data['groups'][group]
            best_so_far = np.maximum.accumulate(np.nan_to_num(values, nan=0))
            x = np.arange(len(best_so_far))
            
            ax.plot(x, best_so_far, color=color, linestyle=linestyle,
                    label=label, linewidth=2)
        
        ax.set_xlabel('API Calls', fontsize=10)
        ax.set_ylabel('Best AOCC', fontsize=10)
        ax.set_title(GROUP_DISPLAY_NAMES[group], fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    # Use the last subplot for legend
    axes[-1].axis('off')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[-1].legend(handles, labels, loc='center', fontsize=12)
    
    plt.suptitle('Convergence by Function Group', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_all_groups_grid.png'), dpi=150)
    plt.close()


def plot_bar_comparison(experiments: List[Dict], output_dir: str):
    """Plot bar chart comparing final best scores."""
    groups = list(FUNCTION_GROUPS.keys())
    n_groups = len(groups)
    n_experiments = len(experiments)
    
    # Calculate bar positions
    bar_width = 0.8 / n_experiments
    x = np.arange(n_groups)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    for i, exp in enumerate(experiments):
        data = exp['data']
        label = exp['label']
        color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
        
        scores = []
        for group in groups:
            group_data = data['groups'][group]
            best = np.nanmax(group_data) if len(group_data) > 0 else 0
            scores.append(best)
        
        offset = (i - n_experiments / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, scores, bar_width * 0.9, 
                     label=label, color=color, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            if score > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{score:.2f}', ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('Function Group', fontsize=12)
    ax.set_ylabel('Best AOCC', fontsize=12)
    ax.set_title('Best Performance by Function Group', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([GROUP_DISPLAY_NAMES[g] for g in groups], rotation=30, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_bar_chart.png'), dpi=150)
    plt.close()


def plot_multimodal_comparison(experiments: List[Dict], output_dir: str):
    """Plot detailed multimodal comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = [
        ('multimodal', 'All Multimodal (f15-f24)'),
        ('multimodal_adequate', 'Multimodal Adequate (f15-f19)'),
        ('multimodal_weak', 'Multimodal Weak (f20-f24)')
    ]
    
    for ax, (metric, title) in zip(axes, metrics):
        for i, exp in enumerate(experiments):
            data = exp['data']
            label = exp['label']
            color = EXPERIMENT_COLORS[i % len(EXPERIMENT_COLORS)]
            linestyle = LINE_STYLES[i % len(LINE_STYLES)]
            
            values = data[metric]
            best_so_far = np.maximum.accumulate(np.nan_to_num(values, nan=0))
            x = np.arange(len(best_so_far))
            
            ax.plot(x, best_so_far, color=color, linestyle=linestyle,
                    label=label, linewidth=2.5)
        
        ax.set_xlabel('API Calls', fontsize=11)
        ax.set_ylabel('Best AOCC', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.6)  # Multimodal typically lower
        ax.legend(fontsize=9)
    
    plt.suptitle('Multimodal Function Performance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_multimodal_detailed.png'), dpi=150)
    plt.close()


def print_comparison_table(experiments: List[Dict], budget: int = 100):
    """Print a comparison table."""
    print("\n" + "="*80)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("="*80)
    
    # Header
    header = f"{'Metric':<35}"
    for exp in experiments:
        header += f" | {exp['label'][:15]:>15}"
    print(header)
    print("-"*80)
    
    # All functions
    row = f"{'All Functions (f1-f24)':<35}"
    for exp in experiments:
        best = np.nanmax(exp['data']['all'])
        row += f" | {best:>15.4f}"
    print(row)
    
    # Multimodal
    row = f"{'Multimodal (f15-f24)':<35}"
    for exp in experiments:
        best = np.nanmax(exp['data']['multimodal'])
        row += f" | {best:>15.4f}"
    print(row)
    
    print("-"*80)
    
    # Per group
    for group in FUNCTION_GROUPS.keys():
        row = f"{GROUP_DISPLAY_NAMES[group]:<35}"
        for exp in experiments:
            best = np.nanmax(exp['data']['groups'][group])
            row += f" | {best:>15.4f}"
        print(row)
    
    print("-"*80)
    
    # Best algorithms
    print("\nBest Algorithms (Overall):")
    for exp in experiments:
        idx, score, name = get_best_algorithm(exp['dir'], budget, 'all')
        print(f"  {exp['label']}: {name} (AOCC: {score:.4f})")
    
    print("\nBest Algorithms (Multimodal):")
    for exp in experiments:
        idx, score, name = get_best_algorithm(exp['dir'], budget, 'multimodal')
        print(f"  {exp['label']}: {name} (AOCC: {score:.4f})")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare multiple experiment results on BBOB functions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Compare two experiments with auto-generated labels
    python compare_experiments.py exp1/ exp2/
    
    # Compare with custom labels
    python compare_experiments.py exp1/ exp2/ exp3/ --labels "Baseline" "GA-LLAMEA" "MADA"
    
    # Save to custom output directory
    python compare_experiments.py exp1/ exp2/ --output comparison_plots/
    
    # Only multimodal comparison
    python compare_experiments.py exp1/ exp2/ --multimodal-only
    
    # Print summary only (no plots)
    python compare_experiments.py exp1/ exp2/ --summary-only
        """
    )
    
    parser.add_argument('exp_dirs', type=str, nargs='+',
                       help='Experiment directories to compare')
    parser.add_argument('--labels', '-l', type=str, nargs='+', default=None,
                       help='Custom labels for each experiment')
    parser.add_argument('--output', '-o', type=str, default='comparison_results',
                       help='Output directory for plots')
    parser.add_argument('--budget', '-b', type=int, default=100,
                       help='Maximum number of evaluations to analyze')
    parser.add_argument('--multimodal-only', '-m', action='store_true',
                       help='Only generate multimodal comparison plots')
    parser.add_argument('--summary-only', '-s', action='store_true',
                       help='Only print summary, no plots')
    
    args = parser.parse_args()
    
    # Validate directories
    valid_dirs = []
    for exp_dir in args.exp_dirs:
        if os.path.isdir(exp_dir):
            valid_dirs.append(exp_dir)
        else:
            print(f"Warning: Directory not found: {exp_dir}")
    
    if len(valid_dirs) < 1:
        print("Error: No valid experiment directories found")
        return
    
    # Generate labels if not provided
    if args.labels and len(args.labels) >= len(valid_dirs):
        labels = args.labels[:len(valid_dirs)]
    else:
        labels = []
        for d in valid_dirs:
            # Extract a short label from directory name
            name = os.path.basename(d.rstrip('/\\'))
            # Try to extract meaningful part
            if 'baseline' in name.lower():
                label = 'Baseline'
            elif 'elitism' in name.lower():
                label = 'Elitism'
            elif 'mada' in name.lower():
                label = 'MADA'
            elif 'ga' in name.lower():
                label = 'GA-LLAMEA'
            else:
                # Use first part of name
                label = name[:30]
            labels.append(label)
    
    # Load data
    print("Loading experiment data...")
    experiments = []
    for exp_dir, label in zip(valid_dirs, labels):
        print(f"  Loading: {label} ({os.path.basename(exp_dir)})")
        data = load_experiment_data(exp_dir, args.budget)
        experiments.append({
            'dir': exp_dir,
            'label': label,
            'data': data
        })
    
    # Print comparison table
    print_comparison_table(experiments, args.budget)
    
    if args.summary_only:
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate plots
    print("Generating comparison plots...")
    
    if args.multimodal_only:
        plot_convergence_comparison(experiments, args.output, 'multimodal')
        plot_multimodal_comparison(experiments, args.output)
    else:
        # All convergence plots
        plot_convergence_comparison(experiments, args.output, 'all')
        plot_convergence_comparison(experiments, args.output, 'multimodal')
        
        # Grid of all groups
        plot_all_groups_comparison(experiments, args.output)
        
        # Bar chart comparison
        plot_bar_comparison(experiments, args.output)
        
        # Detailed multimodal
        plot_multimodal_comparison(experiments, args.output)
    
    print(f"\nPlots saved to: {args.output}")
    print("Files generated:")
    for f in os.listdir(args.output):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
