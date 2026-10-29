"""Streamlit dashboard: model status, risk score, trends, explanations, alert log.

Run:  streamlit run app/dashboard.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from degradation_risk.config import Paths  # noqa: E402
from degradation_risk.pipeline import EVENT_LOG_COLUMNS  # noqa: E402

st.set_page_config(page_title="ML Degradation Risk Monitor", page_icon="📉", layout="wide")
paths = Paths.default()
TABLE, SCORED = paths.processed / "monitoring_table.csv", paths.processed / "scored_bookings.csv.gz"
METRICS, COMPARE, EVENTS = paths.reports / "metrics.json", paths.reports / "model_comparison.csv", paths.processed / "event_log.csv"

STATUS_STYLE = {"normal": ("🟢", "Normal"), "investigate": ("🟠", "Investigate"), "high_risk": ("🔴", "High risk")}
RED, BLUE, GREEN, ORANGE, GREY = "#c0392b", "#1f5f8b", "#2e7d32", "#e67e22", "#7f8c8d"


@st.cache_data(show_spinner=False)
def load_table(mtime: float) -> pd.DataFrame:
    return pd.read_csv(TABLE, parse_dates=["window_start", "window_end"])


@st.cache_data(show_spinner=False)
def load_scored(mtime: float) -> pd.DataFrame:
    return pd.read_csv(SCORED, parse_dates=["arrival_date"])


def load_metrics() -> dict:
    return json.loads(METRICS.read_text())


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def shade(ax, x, mask):
    ax.fill_between(x, 0, 1, where=np.asarray(mask, bool), transform=ax.get_xaxis_transform(),
                    color="#f1948a", alpha=0.25, step="mid", linewidth=0)


# ------------------------------------------------------------------ guard ----
if not TABLE.exists() or not METRICS.exists():
    st.title("📉 ML Model Performance Degradation Risk")
    st.error("No pipeline output found. Run `python scripts/run_pipeline.py` first, then reload this page.")
    st.stop()

win_all = load_table(TABLE.stat().st_mtime)
metrics = load_metrics()
cfg = metrics["config"]
delay = cfg["label_delay"]
f1_ref = metrics["baseline_model"]["f1_ref"]
f1_thr = metrics["baseline_model"]["degradation_f1_threshold"]
model_thr_default = float(metrics["alert_threshold"])

# ---------------------------------------------------------------- sidebar ----
with st.sidebar:
    st.header("Controls")
    ids = win_all["window_id"].tolist()
    start_of = dict(zip(win_all["window_id"], win_all["window_start"].dt.strftime("%Y-%m-%d")))
    as_of = st.select_slider("Replay: view the monitor as of window", options=ids, value=ids[-1],
                             format_func=lambda i: f"#{i} · {start_of[i]}")
    thr = st.slider("Alert threshold", 0.05, 0.95, model_thr_default, 0.01,
                    help="Chosen on validation windows by the pipeline; move it to see the alert burden change.")
    show_split = st.checkbox("Mark validation/test boundaries", value=True)
    st.divider()
    st.caption(f"Window = {cfg['window_days']} days · horizon = next {cfg['horizon']} windows · "
               f"degradation = F1 drop > {cfg['drop_threshold']} · label delay = {delay} window(s)")

win = win_all[win_all["window_id"] <= as_of].reset_index(drop=True)
cur = win.iloc[-1]
risk = float(cur["risk"])
status_key = "high_risk" if risk >= thr else ("investigate" if risk >= 0.5 * thr else "normal")
icon, label = STATUS_STYLE[status_key]

# ----------------------------------------------------------------- header ----
st.title("📉 ML Model Performance Degradation Risk")
st.caption(f"Monitored model: hotel-booking cancellation classifier · data source: **{metrics['data_source']}**")
if metrics.get("synthetic_warning"):
    st.warning(metrics["synthetic_warning"] + " Numbers below are not measured production results.")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Model status", f"{icon} {label}")
c2.metric("Degradation risk", f"{risk:.0%}", help=f"Probability that F1 falls below {f1_thr:.3f} in the next {cfg['horizon']} windows")
c3.metric("Alert threshold", f"{thr:.0%}")
known = win["realized_f1_valid"].where(win["window_id"] <= as_of - delay).dropna()
last_known = float(known.iloc[-1]) if len(known) else float("nan")
c4.metric("Latest known F1", f"{last_known:.3f}", f"{last_known - f1_ref:+.3f} vs reference", delta_color="normal")
c5.metric("Ground truth available through", f"window #{as_of - delay}",
          help=f"Labels arrive {delay} window(s) late; estimated metrics would be flagged as estimates.")

tab_risk, tab_signals, tab_why, tab_ref, tab_eval, tab_log = st.tabs(
    ["Risk & performance", "Monitoring signals", "Why this score", "Reference vs window", "Model evaluation", "Event log"])

# --------------------------------------------------------- risk & performance
with tab_risk:
    x = win["window_start"]
    real = win["realized_f1_valid"].where(win["window_id"] <= as_of - delay)  # only what has arrived
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    shade(a1, x, win["degraded_now"] & (win["window_id"] <= as_of - delay))
    shade(a2, x, win["degraded_now"] & (win["window_id"] <= as_of - delay))
    a1.plot(x, real, color=BLUE, lw=1.2, label="Realized F1")
    a1.axhline(f1_ref, color=GREEN, ls="--", lw=1, label=f"Reference {f1_ref:.3f}")
    a1.axhline(f1_thr, color=ORANGE, ls=":", lw=1.4, label=f"Degradation threshold {f1_thr:.3f}")
    a1.set_ylabel("F1")
    a1.legend(loc="lower left", ncol=3, fontsize=8)
    a2.plot(x, win["risk"], color=RED, lw=1.3, label="Risk probability")
    a2.axhline(thr, color=ORANGE, ls=":", lw=1.4, label=f"Alert threshold {thr:.0%}")
    alerts = win[win["risk"] >= thr]
    a2.scatter(alerts["window_start"], alerts["risk"], s=14, color=RED, zorder=3, label="Alert")
    a2.set_ylim(0, 1.05)
    a2.set_ylabel("Risk")
    a2.legend(loc="upper left", ncol=3, fontsize=8)
    if show_split:
        for sp in ("val", "test"):
            first = win.loc[win["split"] == sp, "window_start"]
            if len(first):
                for ax in (a1, a2):
                    ax.axvline(first.iloc[0], color=GREY, ls="-.", lw=0.9)
                a2.text(first.iloc[0], 1.02, f" {sp} starts", fontsize=8, color=GREY, va="bottom")
    for ax in (a1, a2):
        style(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Shaded = windows whose realized F1 was already below the degradation threshold. "
               "Before the validation line the risk model has seen those windows in training, so judge it on the test segment.")
    in_test = win[win["split"] == "test"]
    if len(in_test):
        n_alert = int((in_test["risk"] >= thr).sum())
        st.write(f"At this threshold: **{n_alert}** alert windows out of **{len(in_test)}** visible test windows "
                 f"({n_alert / len(in_test):.0%} alert rate).")

# ------------------------------------------------------------------ signals
with tab_signals:
    x = win["window_start"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    panels = [
        ("Feature drift (PSI)", [("feature_drift_score", "mean PSI", BLUE), ("max_psi", "max PSI", GREY)], 0.2),
        ("Missing-value rate", [("missing_rate", "missing rate", RED)], None),
        ("Prediction distribution", [("prediction_mean", "mean P(cancel)", GREEN), ("low_confidence_rate", "low-confidence rate", GREY)], None),
        ("Sample volume", [("n_predictions", "predictions per window", BLUE)], None),
    ]
    for ax, (title, cols, ref_line) in zip(axes.ravel(), panels):
        shade(ax, x, win["degraded_now"] & (win["window_id"] <= as_of - delay))
        for col, lab, colr in cols:
            ax.plot(x, win[col], color=colr, lw=1.2, label=lab)
        if ref_line:
            ax.axhline(ref_line, color=ORANGE, ls=":", lw=1, label="PSI 0.2 rule of thumb")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        style(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Drift is a signal, not a verdict: some drift episodes never hurt performance. Compare with the F1 chart.")

# ------------------------------------------------------------------ why
with tab_why:
    st.subheader(f"Signals behind the risk score of window #{as_of}")
    contrib_cols = [c for c in win.columns if c.startswith("contrib_")]
    contrib = cur[contrib_cols].astype(float)
    contrib.index = [c.replace("contrib_", "") for c in contrib.index]
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(8)[::-1]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(top.index, top.values, color=[RED if v > 0 else GREEN for v in top.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Change in risk probability caused by this signal (red = raises risk)")
    style(ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("Occlusion explanation: each signal is replaced by its typical healthy value and the drop in the risk score is measured.")
    feats = metrics["risk_features"]
    snap = pd.DataFrame({"this window": cur[feats].astype(float),
                         "median of all windows": win_all[feats].median()}).round(4)
    st.dataframe(snap, use_container_width=True)

# ------------------------------------------------------------------ reference vs window
with tab_ref:
    if not SCORED.exists():
        st.info("scored_bookings.csv.gz not found; re-run the pipeline.")
    else:
        scored = load_scored(SCORED.stat().st_mtime)
        ref = scored[scored["period"] == "reference"]
        now = scored[scored["window_id"] == as_of]
        options = [c for c in scored.columns if c not in {"window_id", "arrival_date", "period", "is_canceled"}]
        feat = st.selectbox("Feature", options, index=options.index("lead_time") if "lead_time" in options else 0)
        st.write(f"Reference period: **{len(ref):,}** bookings · window #{as_of}: **{len(now):,}** bookings")
        fig, ax = plt.subplots(figsize=(8, 3.8))
        if pd.api.types.is_numeric_dtype(scored[feat]):
            lo, hi = np.nanpercentile(ref[feat], [0.5, 99.5])
            bins = np.linspace(lo, hi, 30)
            ax.hist(ref[feat].dropna(), bins=bins, density=True, alpha=0.5, color=BLUE, label="reference")
            ax.hist(now[feat].dropna(), bins=bins, density=True, alpha=0.5, color=RED, label=f"window #{as_of}")
        else:
            cats = ref[feat].fillna("missing").value_counts(normalize=True).head(10).index
            r = ref[feat].fillna("missing").value_counts(normalize=True).reindex(cats).fillna(0)
            n = now[feat].fillna("missing").value_counts(normalize=True).reindex(cats).fillna(0)
            pos = np.arange(len(cats))
            ax.bar(pos - 0.2, r.values, 0.4, color=BLUE, alpha=0.7, label="reference")
            ax.bar(pos + 0.2, n.values, 0.4, color=RED, alpha=0.7, label=f"window #{as_of}")
            ax.set_xticks(pos, [str(c) for c in cats], rotation=30)
        ax.legend()
        style(ax)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Predictions and outcomes of the reference period vs. the selected monitoring window. "
                   "A single window is small, so expect sampling noise.")

# ------------------------------------------------------------------ evaluation
with tab_eval:
    st.subheader("Chronological evaluation of the risk classifier")
    lt, fa, lab = metrics["lead_time"], metrics["false_alerts"], metrics["labels"]
    e1, e2, e3, e4 = st.columns(4)
    sel = metrics["test_metrics_selected"]
    e1.metric("Selected model", metrics["selected_model"])
    e2.metric("Test PR-AUC", f"{sel['pr_auc']:.3f}", help=f"Prevalence of degradation in test = {sel['prevalence']:.2f}")
    e3.metric("Test precision / recall", f"{sel['precision']:.2f} / {sel['recall']:.2f}")
    e4.metric("False alerts / month", f"{fa['false_alerts_per_month']:.1f}")
    st.write(f"Degradation events in the test period: **{lt['n_events']}**, detected early: **{lt['n_detected_early']}** "
             f"(mean lead time {lt['mean_lead_days']:.1f} days). With so few events, treat these numbers as indicative only.")
    if COMPARE.exists():
        comp = pd.read_csv(COMPARE)
        shown = ["method", "kind", "split", "pr_auc", "roc_auc", "precision", "recall", "recall_at_budget", "brier", "threshold"]
        st.dataframe(comp[shown].round(3), use_container_width=True, hide_index=True)
    fcols = st.columns(2)
    for col, name in zip(fcols, ["pr_curves.png", "calibration.png"]):
        p = paths.figures / name
        if p.exists():
            col.image(str(p))
    p = paths.figures / "signal_importance.png"
    if p.exists():
        st.image(str(p), width=520)
    st.markdown(
        f"**Target:** `degradation_label(t) = 1` if `min(F1(t+1..t+{cfg['horizon']})) < {f1_ref:.3f} - {cfg['drop_threshold']}`. "
        "Features use only information available at time *t*; realized F1 is lagged by the label delay.")

# ------------------------------------------------------------------ event log
with tab_log:
    st.subheader("Alert investigation log")
    log = pd.read_csv(EVENTS, dtype=str).fillna("") if EVENTS.exists() else pd.DataFrame(columns=EVENT_LOG_COLUMNS)
    st.dataframe(log, use_container_width=True, hide_index=True)
    with st.form("add_event", clear_on_submit=True):
        st.write("Add an entry (alerts should trigger investigation, not blind automatic retraining).")
        f1c, f2c = st.columns(2)
        wid = f1c.number_input("Window id", min_value=0, value=int(as_of), step=1)
        reviewer = f2c.text_input("Reviewed by")
        notes = st.text_area("Investigation notes", placeholder="Data quality? pipeline change? label quality? business change?")
        action = st.selectbox("Action taken", ["", "No action - false alarm", "Fixed data pipeline", "Watching",
                                               "Escalated to model owner", "Retraining scheduled"])
        if st.form_submit_button("Save entry"):
            row = {"alert_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"), "window_id": str(int(wid)),
                   "risk_probability": f"{float(win_all.loc[win_all['window_id'] == wid, 'risk'].iloc[0]):.3f}"
                   if (win_all["window_id"] == wid).any() else "",
                   "status": "reviewed", "reviewed_by": reviewer, "investigation_notes": notes, "action_taken": action}
            pd.concat([log, pd.DataFrame([row])], ignore_index=True)[EVENT_LOG_COLUMNS].to_csv(EVENTS, index=False)
            st.success("Saved.")
            st.rerun()
