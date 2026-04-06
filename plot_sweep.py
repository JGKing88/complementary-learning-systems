"""Plot sweep accuracy over iterations.

Queries wandb for all runs in the dist-encoder project, plots val_nav/accuracy
vs iteration index. Runs that improve over previous best are highlighted;
non-improving runs are faint.

Usage:
    python plot_sweep.py [--sweep-id SWEEP_ID] [--output sweep_accuracy.png]
"""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import wandb


def plot_sweep_accuracy(project="generalization-bounds/dist-encoder",
                        sweep_id=None, output="sweep_accuracy.png"):
    api = wandb.Api()

    filters = {}
    if sweep_id:
        filters["sweep"] = sweep_id

    runs = api.runs(project, filters=filters, order="+created_at")

    iterations = []
    accuracies = []
    run_names = []

    for run in runs:
        acc = run.summary.get("val_nav/accuracy")
        if acc is None or np.isnan(acc):
            continue
        iterations.append(len(iterations))
        accuracies.append(acc)
        run_names.append(run.name)

    if not iterations:
        print("No completed runs with val_nav/accuracy found.")
        return

    iterations = np.array(iterations)
    accuracies = np.array(accuracies)

    # Track best so far
    best_so_far = -1.0
    is_improvement = []
    for acc in accuracies:
        if acc > best_so_far:
            is_improvement.append(True)
            best_so_far = acc
        else:
            is_improvement.append(False)
    is_improvement = np.array(is_improvement)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Non-improving runs (faint)
    mask_faint = ~is_improvement
    if mask_faint.any():
        ax.scatter(iterations[mask_faint], accuracies[mask_faint],
                   c='steelblue', alpha=0.15, s=40, zorder=2, label='Non-improving')

    # Improving runs (bold)
    if is_improvement.any():
        ax.scatter(iterations[is_improvement], accuracies[is_improvement],
                   c='crimson', alpha=1.0, s=80, zorder=4, edgecolors='darkred',
                   linewidths=1.0, label='New best')

    # Best-so-far line
    best_line = np.maximum.accumulate(accuracies)
    ax.step(iterations, best_line, where='post', c='crimson', alpha=0.6,
            linewidth=2, zorder=3, label='Best so far')

    # All points connected faintly
    ax.plot(iterations, accuracies, c='steelblue', alpha=0.2, linewidth=0.8, zorder=1)

    ax.set_xlabel('Iteration (sweep run index)', fontsize=12)
    ax.set_ylabel('val_nav/accuracy', fontsize=12)
    ax.set_title(f'Encoder Sweep: Navigation Accuracy\n'
                 f'{len(iterations)} runs, best={best_line[-1]:.3f}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.02, max(1.0, accuracies.max() + 0.05))

    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output}")
    plt.close(fig)

    # Print top 5 runs
    sorted_idx = np.argsort(accuracies)[::-1]
    print(f"\nTop 5 runs:")
    for rank, idx in enumerate(sorted_idx[:5]):
        print(f"  {rank+1}. {run_names[idx]}: acc={accuracies[idx]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-id", type=str, default=None)
    parser.add_argument("--output", type=str, default="sweep_accuracy.png")
    parser.add_argument("--project", type=str, default="generalization-bounds/dist-encoder")
    args = parser.parse_args()

    plot_sweep_accuracy(project=args.project, sweep_id=args.sweep_id, output=args.output)
