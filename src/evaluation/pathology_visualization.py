"""Pathology attention visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_attention_coordinates(coordinates, attention, output_path: str | Path, *, title: str) -> None:
    """Plot genuine model attention values at extracted patch coordinates."""

    coordinates = np.asarray(coordinates)
    attention = np.asarray(attention)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=attention, cmap="magma", s=58, edgecolor="white", linewidth=0.3)
    ax.invert_yaxis()
    ax.set_xlabel("Patch x coordinate")
    ax.set_ylabel("Patch y coordinate")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="Attention weight")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

