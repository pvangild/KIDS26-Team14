"""
event_alignment.py
Team 14 - Multimodal Sleep Disordered Breathing

Fixes a real bug affecting any recording whose measured sample rate isn't
exactly 20.0 Hz (Anna found this on Sept 17 -- see data_quality_exclusions.md,
108 "recovered" files were affected).

THE BUG
-------
events.apneas / events.hypopneas sample indices were generated at export
time assuming a fixed nominal 20.0 Hz sampling rate (per the blueprint:
Index = round(timestamp_sec * 20) + 1). But a subset of files' *actual*
resampled rate deviates from 20.0 Hz (observed range: 16.0-21.33 Hz).

analyze_recordings.py's load_recording()/plot_event() currently do direct
1-based -> 0-based row indexing into `data` with NO rate conversion at all:

    start_index, end_index = (int(value) - 1 for value in pairs[event_index])
    start_time, end_time = time[start_index], time[end_index]

This is only correct if the file's actual sample rate is exactly 20.0 Hz.
For files where it isn't, this silently returns the wrong time window --
confirmed empirically below on ahiData-pats-814664-baseline.mat (measured
rate 21.33 Hz): direct indexing gives 1.1% OriginalDetections overlap
(clearly wrong), while converting the index to seconds using the FIXED
nominal 20.0 Hz and then locating that time in the file's own time array
gives 100.0% overlap (correct).

THE FIX
-------
resolve_event_window() below does the conversion correctly: event index ->
seconds using NOMINAL_SAMPLE_RATE_HZ (never the file's measured rate) ->
locate the corresponding sample in the file's own `time` array via
searchsorted (works regardless of that file's actual sample count/rate).

USAGE
-----
    from analyze_recordings import load_recording, event_pairs
    from event_alignment import resolve_event_window

    time, data, labels, events = load_recording(path)
    pairs = event_pairs(events, "apneas")
    start_time, end_time, start_idx, end_idx = resolve_event_window(time, pairs[0])
    # start_idx/end_idx now index correctly into `data`, regardless of
    # whether this file's actual rate matches the nominal 20.0 Hz.
"""

from __future__ import annotations

import numpy as np

# Event indices in every ahiData-pats-*.mat file were generated assuming
# this rate at export time (per the PATS blueprint's stated formula:
# Index = round(timestamp_sec * NOMINAL_SAMPLE_RATE_HZ) + 1). This must
# stay fixed at 20.0 -- do NOT substitute a file's measured rate here, that
# is exactly the bug this module fixes.
NOMINAL_SAMPLE_RATE_HZ = 20.0


def resolve_event_window(
    time: np.ndarray, event_pair: tuple[int, int]
) -> tuple[float, float, int, int]:
    """
    Correctly convert a 1-based [start_idx, end_idx] event pair (as stored
    in events.apneas / events.hypopneas) into a time window and the
    corresponding row indices into this file's actual `data`/`time` arrays.

    Safe to use even when this file's actual sample rate isn't 20.0 Hz --
    which analyze_recordings.py's current direct-indexing approach is not.

    Returns: (start_time_sec, end_time_sec, start_row_idx, end_row_idx)
    """
    start_idx_1based, end_idx_1based = int(event_pair[0]), int(event_pair[1])

    # Step 1: index -> seconds, using the FIXED nominal rate the index was
    # generated with (not this file's own measured rate).
    start_time = (start_idx_1based - 1) / NOMINAL_SAMPLE_RATE_HZ
    end_time = (end_idx_1based - 1) / NOMINAL_SAMPLE_RATE_HZ

    # Step 2: seconds -> this file's own actual row index, via its real
    # time array (correct regardless of this file's actual sample count).
    start_row = int(np.searchsorted(time, start_time))
    end_row = int(np.searchsorted(time, end_time))
    start_row = min(start_row, len(time) - 1)
    end_row = min(end_row, len(time) - 1)

    return start_time, end_time, start_row, end_row


def check_alignment(time: np.ndarray, data: np.ndarray, labels: list[str], events, event_pairs_fn) -> dict:
    """
    Diagnostic: for a loaded recording, report % of OriginalDetections
    samples that fall within the scored event windows, using the corrected
    conversion. Useful for spot-checking whether a given file needs this
    fix at all (files at exactly 20.0 Hz will show ~100% either way).

    events / event_pairs_fn: pass events + event_pairs from analyze_recordings.py
    """
    od_col = labels.index("OriginalDetections")
    total_overlap = 0
    total_len = 0

    for event_type in ("apneas", "hypopneas"):
        pairs = event_pairs_fn(events, event_type)
        for pair in pairs:
            _, _, start_row, end_row = resolve_event_window(time, pair)
            seg = data[start_row : end_row + 1, od_col]
            total_len += len(seg)
            total_overlap += int(np.sum(seg > 500))

    overlap_pct = 100 * total_overlap / total_len if total_len else float("nan")
    measured_rate = 1 / np.median(np.diff(time))
    return {
        "overlap_pct": overlap_pct,
        "measured_sample_rate_hz": measured_rate,
        "rate_matches_nominal": abs(measured_rate - NOMINAL_SAMPLE_RATE_HZ) < 0.01,
    }


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from analyze_recordings import load_recording, event_pairs

    if len(sys.argv) < 2:
        print("Usage: python event_alignment.py <path-to-recording.mat>")
        sys.exit(1)

    recording_path = Path(sys.argv[1])
    time, data, labels, events = load_recording(recording_path)
    result = check_alignment(time, data, labels, events, event_pairs)

    print(f"{recording_path.name}")
    print(f"  measured sample rate: {result['measured_sample_rate_hz']:.3f} Hz")
    print(f"  matches nominal 20.0 Hz: {result['rate_matches_nominal']}")
    print(f"  OriginalDetections overlap using corrected conversion: {result['overlap_pct']:.1f}%")
