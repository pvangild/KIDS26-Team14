from dataclasses import dataclass
import numpy as np
from scipy.io import loadmat

@dataclass
class Recording:
    time: np.ndarray
    data: np.ndarray
    labels: list
    events: object
    metrics: dict
    study_number: int

def load_recording(filepath):
    mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    signals = mat["signals"]

    time = np.asarray(signals.time).squeeze()
    data = np.asarray(signals.data)
    if data.shape[0] != len(time) and data.shape[1] == len(time):
        data = data.T

    labels = [str(label).strip() for label in np.atleast_1d(signals.labels)]

    return Recording(
        time=time,
        data=data,
        labels=labels,
        events=mat["events"],
        metrics=mat["metrics"] if isinstance(mat["metrics"], dict) else vars(mat["metrics"]),
        study_number=int(mat["study_number"]),
    )
from pathlib import Path

def load_all_recordings(data_dir):
    recordings = []
    failed_files = []

    for filepath in sorted(Path(data_dir).glob("*.mat")):
        try:
            rec = load_recording(filepath)
            recordings.append(rec)
        except Exception as error:
            failed_files.append((filepath.name, str(error)))

    print(f"Loaded {len(recordings)} recordings successfully.")
    if failed_files:
        print(f"Failed to load {len(failed_files)} files:")
        for name, error in failed_files:
            print(f"  {name}: {error}")

    return recordings, failed_files
import pandas as pd

def summarize_recording(rec):
    return {
        "study_number": rec.study_number,
        "ahi": rec.metrics["ahi"],
        "apnea_idx": rec.metrics["apneaIdx"],
        "hypop_idx": rec.metrics["hypopIdx"],
        "recording_hours": rec.time[-1] / 3600,
    }

def build_summary_table(recordings):
    rows = [summarize_recording(rec) for rec in recordings]
    return pd.DataFrame(rows)
if __name__ == "__main__":
    recordings, failed = load_all_recordings("data/raw")
    summary = build_summary_table(recordings)
    print(summary.head())
    summary.to_csv("data/summary.csv", index=False)
    print("Saved summary to data/summary.csv")