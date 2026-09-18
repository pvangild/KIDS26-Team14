import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from data_io import load_recording
from evaluate_events import extract_events_vectorized

st.set_page_config(layout="wide")
st.title("Sleep Apnea Event Detection — Team 14")

val_subjects = pd.read_csv("data/val_subjects.csv")
predictions = pd.read_csv("data/val_predictions_tuned.csv")

tab1, tab2 = st.tabs(["Waveform Explorer", "AHI Accuracy"])

with tab1:
    subject_ids = sorted(val_subjects["study_number"].unique())
    selected = st.selectbox("Select a subject", subject_ids)

    filepath = f"data/raw/ahiData-pats-{selected}-baseline.mat"
    rec = load_recording(filepath)

    subject_preds = predictions[predictions["study_number"] == selected].sort_values("start")
    true_apneas = extract_events_vectorized(subject_preds.rename(columns={"predicted_label": "_", "label": "predicted_label"}), "apnea")
    pred_apneas = extract_events_vectorized(subject_preds, "apnea")
    true_hypops = extract_events_vectorized(subject_preds.rename(columns={"predicted_label": "_", "label": "predicted_label"}), "hypopnea")
    pred_hypops = extract_events_vectorized(subject_preds, "hypopnea")

    channels_to_show = ["airFlow", "SpO2", "thorax"]
    fig = go.Figure()

    for i, ch in enumerate(channels_to_show):
        idx = rec.labels.index(ch)
        raw = rec.data[:, idx]
        normalized = (raw - np.mean(raw)) / (np.std(raw) + 1e-8)  # z-score, avoids divide-by-zero
        fig.add_trace(go.Scatter(x=rec.time, y=normalized + i * 6, name=ch, line=dict(width=1)))

    for start, end in true_apneas:
        fig.add_vrect(x0=rec.time[start], x1=rec.time[end], fillcolor="blue", opacity=0.2, line_width=0)
    for start, end in pred_apneas:
        fig.add_vrect(x0=rec.time[start], x1=rec.time[end], fillcolor="red", opacity=0.2, line_width=0)

    fig.update_layout(height=600, title=f"Subject {selected} — blue=true apnea, red=predicted apnea")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Predicted vs. True AHI (Validation Set)")

    ahi_data = []
    for study_number in val_subjects["study_number"]:
        true_ahi = val_subjects.loc[val_subjects["study_number"] == study_number, "ahi"].values[0]
        subject_preds = predictions[predictions["study_number"] == study_number].sort_values("start")
        apneas = extract_events_vectorized(subject_preds, "apnea")
        hypops = extract_events_vectorized(subject_preds, "hypopnea")
        hours = val_subjects.loc[val_subjects["study_number"] == study_number, "recording_hours"].values[0]
        pred_ahi = (len(apneas) + len(hypops)) / hours
        ahi_data.append({"study_number": study_number, "true_ahi": true_ahi, "predicted_ahi": pred_ahi})

    ahi_df = pd.DataFrame(ahi_data)
    max_val = max(ahi_df["true_ahi"].max(), ahi_df["predicted_ahi"].max())

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=ahi_df["true_ahi"], y=ahi_df["predicted_ahi"], mode="markers", marker=dict(size=10)))
    fig2.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect prediction"))
    fig2.update_layout(xaxis_title="True AHI", yaxis_title="Predicted AHI", height=600)
    st.plotly_chart(fig2, use_container_width=True)

    mae = np.mean(np.abs(ahi_df["true_ahi"] - ahi_df["predicted_ahi"]))
    st.metric("AHI Mean Absolute Error", f"{mae:.2f} events/hour")