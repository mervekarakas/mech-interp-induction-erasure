import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def plot_patching_heatmap(
    heatmap: torch.Tensor,
    n_ctx_half: int,
    title: str = "Activation patching: recovery fraction by (layer, position)",
    save_path: str = None,
):
    """
    Plot a layer x position heatmap of patching recovery.

    Colors:
    - Red/warm: patching here recovers the metric (this location matters)
    - Blue/cool: patching here hurts the metric (negative recovery)
    - White: no effect
    """
    fig, ax = plt.subplots(figsize=(16, 5))

    data = heatmap.numpy()

    # Use a diverging colormap centered at 0
    vmax = max(abs(data.min()), abs(data.max()), 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm, origin="lower")

    # Mark the boundary between first and second half
    ax.axvline(x=n_ctx_half + 0.5, color="black", linestyle="--", alpha=0.7, linewidth=1.5)

    ax.set_xlabel("Token position")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    ax.set_yticks(range(data.shape[0]))

    plt.colorbar(im, ax=ax, label="Recovery fraction (0=no help, 1=full recovery)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()


def plot_head_patching_heatmap(
    heatmap: torch.Tensor,
    title: str = "Head patching: recovery fraction by (layer, head)",
    save_path: str = None,
):
    """
    Plot a layer x head heatmap. Each cell shows how much patching
    that single head's output (across all positions) recovers the metric.

    Big red squares = heads that matter for the behavior.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    data = heatmap.numpy()
    vmax = max(abs(data.min()), abs(data.max()), 0.01)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", norm=norm, origin="lower")

    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title(title)
    ax.set_xticks(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))

    # Add text annotations for strong values
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if abs(data[i, j]) > 0.02:
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, label="Recovery fraction")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    plt.show()
