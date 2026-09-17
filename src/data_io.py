from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import numpy as np
from scipy.io import loadmat
from excluded_subjects import EXCLUDED_SUBJECTS

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

def split_train_val(summary_df, val_fraction=0.2, random_state=42):
    severity_bins = pd.cut(
        summary_df["ahi"],
        bins=[-0.01, 1, 5, 10, float("inf")],
        labels=["normal", "mild", "moderate", "severe"],
    )
    train_df, val_df = train_test_split(
        summary_df,
        test_size=val_fraction,
        random_state=random_state,
        stratify=severity_bins,
    )
    return train_df, val_df

def make_windows(rec, window_seconds=30, step_seconds=15):
    fs = 1 / np.median(np.diff(rec.time))  # sampling rate, ~20 Hz
    window_size = int(window_seconds * fs)
    step_size = int(step_seconds * fs)

    apnea_pairs = np.atleast_2d(np.asarray(rec.events.apneas)) if np.asarray(rec.events.apneas).size else np.empty((0, 2))
    hypop_pairs = np.atleast_2d(np.asarray(rec.events.hypopneas)) if np.asarray(rec.events.hypopneas).size else np.empty((0, 2))

    windows = []
    n_samples = rec.data.shape[0]

    for start in range(0, n_samples - window_size, step_size):
        end = start + window_size
        label = "normal"

        for s, e in apnea_pairs:
            if not (e < start or s > end):
                label = "apnea"
                break
        if label == "normal":
            for s, e in hypop_pairs:
                if not (e < start or s > end):
                    label = "hypopnea"
                    break

        windows.append({
            "study_number": rec.study_number,
            "start": start,
            "end": end,
            "label": label,
        })

    return pd.DataFrame(windows)

def extract_window_features(rec, window_seconds=30, step_seconds=15):
    fs = 1 / np.median(np.diff(rec.time))
    window_size = int(window_seconds * fs)
    step_size = int(step_seconds * fs)

    apnea_pairs = np.atleast_2d(np.asarray(rec.events.apneas)) if np.asarray(rec.events.apneas).size else np.empty((0, 2))
    hypop_pairs = np.atleast_2d(np.asarray(rec.events.hypopneas)) if np.asarray(rec.events.hypopneas).size else np.empty((0, 2))

    rows = []
    n_samples = rec.data.shape[0]

    for start in range(0, n_samples - window_size, step_size):
        end = start + window_size
        label = "normal"

        for s, e in apnea_pairs:
            if not (e < start or s > end):
                label = "apnea"
                break
        if label == "normal":
            for s, e in hypop_pairs:
                if not (e < start or s > end):
                    label = "hypopnea"
                    break

        window_data = rec.data[start:end, :]
        row = {"study_number": rec.study_number, "start": start, "end": end, "label": label}
        for i, channel_name in enumerate(rec.labels):
            channel_values = window_data[:, i]
            row[f"{channel_name}_mean"] = np.mean(channel_values)
            row[f"{channel_name}_std"] = np.std(channel_values)
            row[f"{channel_name}_min"] = np.min(channel_values)
            row[f"{channel_name}_max"] = np.max(channel_values)
            row[f"{channel_name}_range"] = np.max(channel_values) - np.min(channel_values)
            row[f"{channel_name}_slope"] = np.polyfit(np.arange(len(channel_values)), channel_values, 1)[0]

        rows.append(row)

    return pd.DataFrame(rows)

def make_windows_for_subjects(subject_ids, data_dir="data/raw", window_seconds=30, step_seconds=15):
    all_windows = []
    for study_number in subject_ids:
        filepath = Path(data_dir) / f"ahiData-pats-{study_number}-baseline.mat"
        try:
            rec = load_recording(filepath)
            windows_df = extract_window_features(rec, window_seconds, step_seconds)
            all_windows.append(windows_df)
        except Exception as error:
            print(f"Skipping {study_number}: {error}")

    return pd.concat(all_windows, ignore_index=True)

if __name__ == "__main__":
    recordings, failed = load_all_recordings("data/raw")
    summary = build_summary_table(recordings)
    summary = summary[~summary["study_number"].isin(EXCLUDED_SUBJECTS)]
    print(f"After exclusions: {len(summary)} clean subjects remain")
    summary.to_csv("data/summary.csv", index=False)
    train_df, val_df = split_train_val(summary)

    print(f"Train subjects: {len(train_df)}, Validation subjects: {len(val_df)}")
    train_df.to_csv("data/train_subjects.csv", index=False)
    val_df.to_csv("data/val_subjects.csv", index=False)

    print("\nTrain AHI stats:")
    print(train_df["ahi"].describe())
    print("\nValidation AHI stats:")
    print(val_df["ahi"].describe())

    import matplotlib.pyplot as plt

    plt.hist(train_df["ahi"], bins=20, alpha=0.5, label="Train")
    plt.hist(val_df["ahi"], bins=20, alpha=0.5, label="Validation")
    plt.xlabel("AHI")
    plt.ylabel("Count")
    plt.legend()
    plt.title("AHI distribution: Train vs Validation")
    plt.show()

    train_windows = make_windows_for_subjects(train_df["study_number"], window_seconds=30, step_seconds=15)
    print("Train windows:", train_windows["label"].value_counts())
    train_windows.to_csv("data/train_windows.csv", index=False)

    val_windows = make_windows_for_subjects(val_df["study_number"], window_seconds=30, step_seconds=15)
    print("Validation windows:", val_windows["label"].value_counts())
    val_windows.to_csv("data/val_windows.csv", index=False)