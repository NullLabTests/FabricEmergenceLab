# add comment block describing the experiment
"""This script visualizes the training and evaluation of a Predictive Coding Network (PCN) on a specified dataset (MNIST)."""

import numpy as np
import plotly.graph_objects as go
from plotly import colors


# Plot
def plot_energy_history_interactive(
    energy_history, title="Batch-averaged Energy Trajectories"
):
    """
    Interactive energy trajectories with Plotly.
    - energy_history: list of epochs; each epoch is list of batch energy lists.
    - infer_steps: number of inference steps.
    Hover to see Epoch, Batch, Step, Energy.
    """
    num_epochs = len(energy_history)
    # Sample one color per epoch from Viridis
    epoch_colors = colors.sample_colorscale(
        colors.sequential.Viridis,
        [i / (num_epochs - 1) if num_epochs > 1 else 0 for i in range(num_epochs)],
    )

    fig = go.Figure()
    for epoch_idx, epoch_energies in enumerate(energy_history):
        color = epoch_colors[epoch_idx]
        for batch_idx, batch_vals in enumerate(epoch_energies):
            steps = list(range(len(batch_vals)))
            # Attach epoch and batch metadata for hover
            customdata = [[epoch_idx + 1, batch_idx + 1] for _ in steps]
            fig.add_trace(
                go.Scatter(
                    x=steps,
                    y=np.log10(batch_vals),
                    mode="lines",
                    line=dict(color=color, width=1),
                    hovertemplate=(
                        "Epoch %{customdata[0]}<br>"
                        "Batch %{customdata[1]}<br>"
                        "Step %{x}<br>"
                        "Energy %{y:.4f}<extra></extra>"
                    ),
                    customdata=customdata,
                    showlegend=False,
                )
            )

    fig.update_layout(
        title=title, xaxis_title="Inference Step t", yaxis_title="Log10 Energy"
    )

    # Optional: log-scale y-axis if needed
    # fig.update_yaxes(type='log')

    fig.show()


# %%
def plot_epoch_avg_interactive(
    energy_history, title="Batch-Averaged Energy Trajectories (Mean ± 1-std)"
):
    """
    Interactive per-epoch mean ±1-std energy trajectories using Plotly.
    - energy_history: list of epochs; each epoch is list of batch energy trajectories.
    - infer_steps: number of inference steps.
    Hover to see Epoch, Step, Mean Energy.
    """
    num_epochs = len(energy_history)
    epoch_colors = colors.sample_colorscale(
        colors.sequential.Viridis,
        [i / (num_epochs - 1) if num_epochs > 1 else 0 for i in range(num_epochs)],
        colortype="hex",  # Makes each color something like '#440154'
    )

    epoch_colors = [
        (
            c
            if isinstance(c, str)
            else "#{0:02x}{1:02x}{2:02x}".format(*(int(round(255 * v)) for v in c))
        )
        for c in epoch_colors
    ]

    fig = go.Figure()

    for epoch_idx, epoch_batches in enumerate(energy_history):
        arr = np.array(epoch_batches)  # shape (n_batches, total_steps+1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        steps = list(range(len(mean)))
        color = epoch_colors[epoch_idx]
        # Upper bound (mean + std)
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean + std,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Lower bound (mean - std) with fill
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean - std,
                mode="lines",
                fill="tonexty",
                # Convert hex '#rrggbb' → rgba(r,g,b,0.2)
                fillcolor=f"rgba({int(color[1:3],16)},"
                f"{int(color[3:5],16)},"
                f"{int(color[5:7],16)},0.2)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Mean line
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mean,
                mode="lines",
                line=dict(color=color, width=2),
                name=f"Epoch {epoch_idx+1}",
                hovertemplate=(
                    f"Epoch {epoch_idx+1}<br>"
                    "Step %{x}<br>"
                    "Energy %{y:.4f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(title=title, xaxis_title="Inference Step t", yaxis_title="Energy")
    # Optional: log-scale y-axis if needed
    # fig.update_yaxes(type='log')
    fig.show()


def plot_train_val_metric(
    train_result: list, val_result: list, yaxis_title="Metric Name", logy=False
):
    """
    Plot training and validation metrics over epochs using Plotly.
    """
    train_color = "#440154"
    val_color = "#053061"
    if len(train_result) != len(val_result):
        raise ValueError("train_result and val_result must have the same length.")
    fig = go.Figure()
    # Add line plots to the figure
    fig.add_trace(
        go.Scatter(
            x=list(range(len(train_result))),
            y=train_result,
            mode="lines",
            line=dict(color=train_color, width=1),
            name="Train",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(range(len(val_result))),
            y=val_result,
            mode="lines",
            line=dict(color=val_color, width=2),
            name="Val",
        )
    )
    fig.update_layout(
        title="Training and Validation Metric Over Epochs",
        xaxis_title="Epoch",
        yaxis_title=yaxis_title,
        legend=dict(
            title="Split",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
        ),
    )
    # Optional: log-scale y-axis if needed
    if logy:
        fig.update_yaxes(type="log")

    fig.show()


def unflatten_images(images, image_size=(3, 32, 32), normalize=True):
    """Unflatten images and tile them vertically for display.

    Args:
        images: Array of shape (batch, flat) or (batch, C, H, W).
        image_size: Tuple of (channels, height, width).
        normalize: Whether to normalize pixel values to [0, 1].

    Returns:
        NumPy array of shape (H, W*batch, 3) suitable for plotly.
    """
    img = np.asarray(images).reshape(-1, *image_size)  # (batch, C, H, W)
    if normalize:
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    img = (255 * img).astype(np.uint8)
    if image_size[0] == 1:
        # Replicate grayscale to RGB
        img = np.repeat(img, 3, axis=1)
    # Tile images vertically: concatenate along height axis
    img = np.concatenate([img[i, :, :, :] for i in range(img.shape[0])], axis=1)
    img = np.transpose(img, (1, 2, 0))  # (C, H, W) -> (H, W, C) for plotly
    return img
