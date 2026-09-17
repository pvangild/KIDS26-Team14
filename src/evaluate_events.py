import pandas as pd
import numpy as np


def extract_events_vectorized(df, target_label):
    """Convert a sorted, single-subject windows dataframe into a list of
    (start, end) events for the given label, using vectorized operations."""
    is_target = (df["predicted_label"] == target_label).values
    if not is_target.any():
        return []

    starts = df["start"].values
    ends = df["end"].values

    change = np.diff(is_target.astype(int))
    event_start_idx = np.where(change == 1)[0] + 1
    if is_target[0]:
        event_start_idx = np.insert(event_start_idx, 0, 0)

    event_end_idx = np.where(change == -1)[0]
    if is_target[-1]:
        event_end_idx = np.append(event_end_idx, len(is_target) - 1)

    return list(zip(starts[event_start_idx], ends[event_end_idx]))


def match_events(true_events, predicted_events, tolerance_samples):
    matched_true = set()
    matched_pred = set()

    for i, (true_start, true_end) in enumerate(true_events):
        padded_start = true_start - tolerance_samples
        padded_end = true_end + tolerance_samples

        for j, (pred_start, pred_end) in enumerate(predicted_events):
            if j in matched_pred:
                continue
            overlaps = not (pred_end < padded_start or pred_start > padded_end)
            if overlaps:
                matched_true.add(i)
                matched_pred.add(j)
                break

    true_positives = len(matched_true)
    false_negatives = len(true_events) - true_positives
    false_positives = len(predicted_events) - len(matched_pred)

    return true_positives, false_positives, false_negatives


def compute_event_f1(predictions_df, label, tolerance_seconds=7.5, fs=20):
    tolerance_samples = tolerance_seconds * fs
    total_tp, total_fp, total_fn = 0, 0, 0

    grouped = predictions_df.sort_values("start").groupby("study_number")

    for study_number, subject_df in grouped:
        true_df = subject_df.drop(columns=["predicted_label"]).rename(columns={"label": "predicted_label"})
        true_events = extract_events_vectorized(true_df, label)
        predicted_events = extract_events_vectorized(subject_df, label)

        tp, fp, fn = match_events(true_events, predicted_events, tolerance_samples)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1


def compute_ahi_mae(predictions_df, val_df):
    results = []
    grouped = predictions_df.sort_values("start").groupby("study_number")
    val_lookup = val_df.set_index("study_number")

    for study_number, subject_df in grouped:
        recording_hours = val_lookup.loc[study_number, "recording_hours"]
        true_ahi = val_lookup.loc[study_number, "ahi"]

        predicted_apneas = extract_events_vectorized(subject_df, "apnea")
        predicted_hypopneas = extract_events_vectorized(subject_df, "hypopnea")
        predicted_ahi = (len(predicted_apneas) + len(predicted_hypopneas)) / recording_hours

        results.append({
            "study_number": study_number,
            "true_ahi": true_ahi,
            "predicted_ahi": predicted_ahi,
        })

    ahi_df = pd.DataFrame(results)
    mae = np.mean(np.abs(ahi_df["true_ahi"] - ahi_df["predicted_ahi"]))
    return mae, ahi_df

def find_best_threshold(predictions_df, label, prob_column, thresholds=np.arange(0.1, 0.6, 0.05)):
    best_threshold = None
    best_f1 = -1

    for threshold in thresholds:
        temp_df = predictions_df.copy()
        temp_df["predicted_label"] = np.where(
            temp_df[prob_column] > threshold, label, "not_" + label
        )
        # Also relabel true labels the same way for a fair comparison
        temp_df["label"] = np.where(temp_df["label"] == label, label, "not_" + label)

        precision, recall, f1 = compute_event_f1(temp_df, label)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1

def find_best_threshold_pair(predictions_df, apnea_thresholds=np.arange(0.05, 0.95, 0.05), hypop_thresholds=np.arange(0.05, 0.95, 0.05)):
    best_combo = None
    best_combined_f1 = -1

    for apnea_t in apnea_thresholds:
        for hypop_t in hypop_thresholds:
            temp_df = predictions_df.copy()
            temp_df["predicted_label"] = np.where(
                temp_df["prob_apnea"] > apnea_t, "apnea",
                np.where(temp_df["prob_hypopnea"] > hypop_t, "hypopnea", "normal")
            )

            _, _, apnea_f1 = compute_event_f1(temp_df, "apnea")
            _, _, hypop_f1 = compute_event_f1(temp_df, "hypopnea")
            combined_f1 = (apnea_f1 + hypop_f1) / 2

            if combined_f1 > best_combined_f1:
                best_combined_f1 = combined_f1
                best_combo = (apnea_t, hypop_t, apnea_f1, hypop_f1)

    return best_combo

if __name__ == "__main__":
    predictions_df = pd.read_csv("data/val_predictions.csv")

    apnea_precision, apnea_recall, apnea_f1 = compute_event_f1(predictions_df, "apnea")
    print(f"Apnea event-level F1: {apnea_f1:.3f} (precision={apnea_precision:.3f}, recall={apnea_recall:.3f})")

    hypop_precision, hypop_recall, hypop_f1 = compute_event_f1(predictions_df, "hypopnea")
    print(f"Hypopnea event-level F1: {hypop_f1:.3f} (precision={hypop_precision:.3f}, recall={hypop_recall:.3f})")

    val_df = pd.read_csv("data/val_subjects.csv")
    mae, ahi_df = compute_ahi_mae(predictions_df, val_df)
    print(f"\nAHI MAE: {mae:.2f} events/hour")
    print(ahi_df.head(10))

    best_apnea_t, best_hypop_t, best_apnea_f1, best_hypop_f1 = find_best_threshold_pair(predictions_df)
    print(f"\nBest joint thresholds: apnea={best_apnea_t:.2f} (F1={best_apnea_f1:.3f}), hypopnea={best_hypop_t:.2f} (F1={best_hypop_f1:.3f})")

    predictions_df["predicted_label"] = np.where(
        predictions_df["prob_apnea"] > best_apnea_t, "apnea",
        np.where(predictions_df["prob_hypopnea"] > best_hypop_t, "hypopnea", "normal")
    )
    predictions_df.to_csv("data/val_predictions_tuned.csv", index=False)

    final_mae, final_ahi_df = compute_ahi_mae(predictions_df, val_df)
    print(f"\nFinal tuned AHI MAE: {final_mae:.2f} events/hour")