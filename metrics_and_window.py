"""
metrics_and_window.py
Team 14 - Multimodal Sleep Disordered Breathing

Small additions on top of analyze_recordings.py:

1. load_metrics()  -> pulls out apneaIdx / hypopIdx / ahi (not currently
   extracted anywhere), so we can compare our model's predicted AHI against
   the clinician-derived ground truth (the challenge's headline metric).

2. plot_window()   -> plots an arbitrary fixed time window across all
   channels, rather than only a window centered on one scored event. Useful
   for general EDA (e.g. "what does 5 minutes of normal breathing look
   like?") without needing a specific apnea/hypopnea to anchor on.

Both reuse load_recording() from analyze_recordings.py rather than
re-parsing the .mat file, so there's only one place that knows how to read
these files.

Usage:
    from analyze_recordings import load_recording
    from metrics_and_window import load_metrics, plot_window

    metrics = load_metrics("ahiData-pats-805603-baseline.mat")
    print(metrics)  # {'apneaIdx': 0.0, 'hypopIdx': 0.308, 'ahi': 0.308}

    plot_window("ahiData-pats-805603-baseline.mat", start_sec=1000, duration_sec=300)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from analyze_recordings import load_recording


def load_metrics(recording_path: Path) -> dict[str, float]:
    """Extract the summary AHI metrics from a recording's .mat file.

    Returns a dict with keys apneaIdx, hypopIdx, ahi (all events/hour).
    """
    mat = loadmat(recording_path, squeeze_me=True, struct_as_record=False)
    try:
        metrics = mat["metrics"]
        return {
            "apneaIdx": float(metrics.apneaIdx),
            "hypopIdx": float(metrics.hypopIdx),
            "ahi": float(metrics.ahi),
        }
    except (AttributeError, KeyError) as error:
        raise ValueError(
            f"{Path(recording_path).name} does not contain the expected metrics struct."
        ) from error


def plot_window(recording_path: Path, start_sec: float, duration_sec: float) -> None:
    """Plot every channel across a fixed [start_sec, start_sec + duration_sec] window.

    Unlike analyze_recordings.plot_event, this isn't tied to a specific
    scored apnea/hypopnea -- pick any window you want to inspect.
    """
    time, data, labels, _ = load_recording(Path(recording_path))

    end_sec = start_sec + duration_sec
    mask = (time >= start_sec) & (time <= end_sec)
    display_time = time[mask]
    display_data = data[mask]

    figure, axes = plt.subplots(len(labels), 1, figsize=(16, 2.1 * len(labels)), sharex=True)
    axes = np.atleast_1d(axes)
    for column, (axis, label) in enumerate(zip(axes, labels)):
        values = display_data[:, column]
        if label == "sleepStages":
            axis.step(display_time, values, where="post")
            axis.set_yticks([0, 1, 2, 3, 5])
            axis.set_yticklabels(["Wake", "N1", "N2", "N3", "REM"])
        elif label == "OriginalDetections":
            axis.fill_between(display_time, 0, (values > 0).astype(float), step="post", alpha=0.5)
            axis.set_ylim(-0.05, 1.05)
            axis.set_yticks([0, 1])
            axis.set_yticklabels(["Normal", "Event"])
        else:
            axis.plot(display_time, values, linewidth=1)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("Time (seconds)")
    figure.suptitle(f"{Path(recording_path).name}: {start_sec:.0f}-{end_sec:.0f} seconds")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python metrics_and_window.py <path-to-recording.mat>")
        sys.exit(1)

    metrics = load_metrics(sys.argv[1])
    print(f"apneaIdx={metrics['apneaIdx']:.3f}  hypopIdx={metrics['hypopIdx']:.3f}  ahi={metrics['ahi']:.3f}")
