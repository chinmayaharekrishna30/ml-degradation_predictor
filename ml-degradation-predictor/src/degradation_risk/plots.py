"""Static figures written by the pipeline (the dashboard draws its own live ones)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

C = {"risk": "#c0392b", "f1": "#1f5f8b", "ref": "#2e7d32", "thr": "#e67e22", "grey": "#7f8c8d", "shade": "#f1948a"}


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


def shade(ax, x, mask, color=C["shade"], alpha=0.25):
    ax.fill_between(x, 0, 1, where=np.asarray(mask, bool), transform=ax.get_xaxis_transform(),
                    color=color, alpha=alpha, step="mid", linewidth=0)


def risk_vs_f1(win, f1_ref, f1_thr, risk_thr, path: Path):
    x = win["window_start"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [1, 1]})
    shade(a1, x, win["degraded_now"])
    shade(a2, x, win["degraded_now"])
    a1.plot(x, win["realized_f1"], color=C["f1"], lw=1.2, label="Realized F1 (per window)")
    a1.axhline(f1_ref, color=C["ref"], ls="--", lw=1, label=f"Reference F1 = {f1_ref:.3f}")
    a1.axhline(f1_thr, color=C["thr"], ls=":", lw=1.3, label=f"Degradation threshold = {f1_thr:.3f}")
    a1.set_ylabel("F1 of cancellation model")
    a1.legend(loc="lower left", fontsize=8, ncol=3)
    a1.set_title("Realized performance (shaded = actually degraded)")
    a2.plot(x, win["risk"], color=C["risk"], lw=1.2, label="Predicted degradation risk")
    a2.axhline(risk_thr, color=C["thr"], ls=":", lw=1.3, label=f"Alert threshold = {risk_thr:.2f}")
    for split, ls in (("val", "-"), ("test", "-")):
        first = win.loc[win["split"] == split, "window_start"]
        if len(first):
            for ax in (a1, a2):
                ax.axvline(first.iloc[0], color=C["grey"], ls="-.", lw=0.9)
            a2.text(first.iloc[0], 1.02, f" {split} starts", fontsize=8, color=C["grey"], va="bottom")
    a2.set_ylim(0, 1.05)
    a2.set_ylabel("Risk probability")
    a2.legend(loc="upper left", fontsize=8, ncol=2)
    a2.set_title("Risk score from monitoring signals (out-of-sample after the val/test lines)")
    for ax in (a1, a2):
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def drift_timeline(win, path: Path):
    x = win["window_start"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    for ax in axes:
        shade(ax, x, win["degraded_now"])
        _style(ax)
    axes[0].plot(x, win["feature_drift_score"], color=C["f1"], label="mean PSI")
    axes[0].plot(x, win["max_psi"], color=C["grey"], lw=0.9, label="max PSI")
    axes[0].axhline(0.2, color=C["thr"], ls=":", label="PSI 0.2 rule of thumb")
    axes[0].set_ylabel("Feature drift")
    axes[0].legend(fontsize=8, ncol=3)
    axes[1].plot(x, win["missing_rate"], color=C["risk"])
    axes[1].set_ylabel("Missing rate")
    axes[2].plot(x, win["prediction_mean"], color=C["ref"], label="mean predicted P(cancel)")
    axes[2].plot(x, win["low_confidence_rate"], color=C["grey"], label="low-confidence rate")
    axes[2].legend(fontsize=8, ncol=2)
    axes[2].set_ylabel("Predictions")
    axes[0].set_title("Monitoring signals over time (shaded = actually degraded)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def pr_curves(y, scores: dict, path: Path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, s in scores.items():
        if np.sum(y) == 0:
            continue
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, lw=1.5, label=name)
    ax.axhline(np.mean(y), color=C["grey"], ls=":", label=f"prevalence = {np.mean(y):.2f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall on the untouched test windows")
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def calibration_plot(table, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], color=C["grey"], ls="--", label="perfect calibration")
    ax.plot(table["mean_predicted"], table["observed_rate"], "o-", color=C["risk"], label="model")
    for _, r in table.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_predicted"], r["observed_rate"]), fontsize=7,
                    xytext=(4, -9), textcoords="offset points")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed degradation rate")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def signal_importance(mean_abs_contrib, path: Path):
    s = mean_abs_contrib.sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(s.index, s.values, color=C["f1"])
    ax.set_xlabel("Mean |contribution to risk score| on test windows")
    ax.set_title("Which monitoring signals drive the risk score")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
