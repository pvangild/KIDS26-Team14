"""Summarize sleep-event recordings and optionally plot one event.

Example:
    python analyze_recordings.py --data-dir recordings --plot-apnea --file 2 --event 3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize apnea and hypopnea events in MATLAB recordings."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Folder containing .mat recording files.",
    )
    parser.add_argument(
        "--file",
        type=int,
        help="1-based position of the recording in alphabetical filename order.",
    )
    parser.add_argument(
        "--event",
        type=int,
        help="1-based event number to plot within the selected event type.",
    )
    plot_group = parser.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plot-apnea", action="store_true", help="Plot an apnea event."
    )
    plot_group.add_argument(
        "--plot-hypopnea", action="store_true", help="Plot a hypopnea event."
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path("event_summary.md"),
        help="Markdown file to create (default: event_summary.md).",
    )
    parser.add_argument(
        "--padding-seconds",
        type=float,
        default=30.0,
        help="Seconds to show before and after a plotted event (default: 30).",
    )
    arguments = parser.parse_args()

    plotting_requested = arguments.plot_apnea or arguments.plot_hypopnea
    if plotting_requested and (arguments.file is None or arguments.event is None):
        parser.error("--file and --event are required when requesting a plot.")
    if not plotting_requested and (arguments.file is not None or arguments.event is not None):
        parser.error("--file and --event require --plot-apnea or --plot-hypopnea.")
    if arguments.file is not None and arguments.file < 1:
        parser.error("--file must be at least 1.")
    if arguments.event is not None and arguments.event < 1:
        parser.error("--event must be at least 1.")
    return arguments


def load_recording(recording_path: Path) -> tuple[np.ndarray, np.ndarray, list[str], object]:
    mat = loadmat(recording_path, squeeze_me=True, struct_as_record=False)
    try:
        signals = mat["signals"]
        events = mat["events"]
        time = np.asarray(signals.time).squeeze()
        data = np.asarray(signals.data)
        labels = [str(label).strip() for label in np.atleast_1d(signals.labels)]
    except (AttributeError, KeyError) as error:
        raise ValueError(f"{recording_path.name} does not match the expected recording format.") from error

    if data.shape[0] != len(time) and data.shape[1] == len(time):
        data = data.T
    if data.shape[0] != len(time):
        raise ValueError(f"{recording_path.name} has incompatible time and signal dimensions.")
    return time, data, labels, events


def event_pairs(events: object, event_type: str) -> np.ndarray:
    values = np.asarray(getattr(events, event_type)).squeeze()
    if values.size == 0:
        return np.empty((0, 2), dtype=int)
    if values.ndim == 1:
        if values.size != 2:
            raise ValueError(f"{event_type} must contain start/end sample-index pairs.")
        return values.reshape(1, 2)
    if values.shape[1] < 2:
        raise ValueError(f"{event_type} must contain start/end sample-index pairs.")
    return values[:, :2]


def write_summary(recording_paths: list[Path], summary_path: Path) -> None:
    lines = ["# Sleep Event Summary", "", "| File | Apneas | Hypopneas |", "| --- | ---: | ---: |"]
    for recording_path in recording_paths:
        _, _, _, events = load_recording(recording_path)
        apnea_count = len(event_pairs(events, "apneas"))
        hypopnea_count = len(event_pairs(events, "hypopneas"))
        lines.append(f"| {recording_path.name} | {apnea_count} | {hypopnea_count} |")

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_event(
    recording_path: Path,
    event_type: str,
    event_number: int,
    padding_seconds: float,
) -> None:
    time, data, labels, events = load_recording(recording_path)
    pairs = event_pairs(events, event_type)
    event_index = event_number - 1
    if event_index >= len(pairs):
        raise IndexError(
            f"{recording_path.name} contains {len(pairs)} {event_type}; "
            f"event {event_number} is unavailable."
        )

    start_index, end_index = (int(value) - 1 for value in pairs[event_index])
    if not (0 <= start_index < len(time) and 0 <= end_index < len(time)):
        raise IndexError(f"{event_type} {event_number} has indices outside the recording.")

    start_time, end_time = time[start_index], time[end_index]
    mask = (time >= start_time - padding_seconds) & (time <= end_time + padding_seconds)
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
        axis.axvspan(start_time, end_time, alpha=0.15)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)

    axes[-1].set_xlabel("Time (seconds)")
    figure.suptitle(
        f"{recording_path.name}: {event_type[:-1].title()} {event_number} "
        f"({start_time:.1f}-{end_time:.1f} s, {end_time - start_time:.1f} s)"
    )
    figure.tight_layout()
    plt.show()


def main() -> None:
    arguments = parse_arguments()
    recording_paths = sorted(arguments.data_dir.glob("*.mat"))
    if not recording_paths:
        raise FileNotFoundError(f"No .mat files found in {arguments.data_dir}.")

    write_summary(recording_paths, arguments.summary_file)
    print(f"Wrote summary for {len(recording_paths)} recordings to {arguments.summary_file}.")

    if arguments.plot_apnea or arguments.plot_hypopnea:
        file_index = arguments.file - 1
        if file_index >= len(recording_paths):
            raise IndexError(
                f"--file {arguments.file} is unavailable; only {len(recording_paths)} recordings were found."
            )
        event_type = "apneas" if arguments.plot_apnea else "hypopneas"
        plot_event(recording_paths[file_index], event_type, arguments.event, arguments.padding_seconds)


if __name__ == "__main__":
    main()