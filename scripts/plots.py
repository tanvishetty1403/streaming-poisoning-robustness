"""Turn results/metrics.csv into the summary tables and the demo figures.

Writes to results/: summary.csv, fig_curves.png, fig_asr_recovery.png, fig_detection.png

    python scripts/plots.py
    python scripts/plots.py --curve-attack targeted_flip --curve-budget 0.5 --curve-timing burst
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

KEYS = ["attack", "budget", "timing"]
LAST_N = 3   # final metric = mean over the last N windows (single windows are noisy)


def final_value(df, col):
    """Per run: mean of the last LAST_N windows. Then mean/std over seeds."""
    per_run = df.sort_values("window").groupby("run_id").tail(LAST_N).groupby("run_id")[col].mean()
    return per_run.mean(), per_run.std(ddof=0)


def summarize(m):
    clean = m[(m["attack"] == "none") & (m["defense"] == "off")]
    rows = []
    for (atk, b, t), g in m[m["attack"] != "none"].groupby(KEYS):
        att, dfn = g[g["defense"] == "off"], g[g["defense"] == "on"]
        if att.empty or dfn.empty or clean.empty:
            continue
        for metric in ("recall", "pr_auc"):
            c, a_, d_ = (final_value(x, metric)[0] for x in (clean, att, dfn))
            asr = (c - a_) / c if c else np.nan
            rec = (d_ - a_) / (c - a_) if (c - a_) > 1e-6 else np.nan   # undefined if the attack did no damage
            rows.append(dict(attack=atk, budget=b, timing=t, metric=metric, clean=c, attacked=a_,
                             defended=d_, ASR=asr, recovery=rec))
    s = pd.DataFrame(rows)
    # detection numbers from the defended runs
    det = m[(m["defense"] == "on") & (m["attack"] != "none")].copy()
    d = (det.groupby(KEYS).agg(det_precision=("det_precision", "mean"), det_recall=("det_recall", "mean"),
                               det_fpr=("det_fpr", "mean"), time_to_detect=("time_to_detect", "mean"),
                               backdoor_hit_rate=("backdoor_hit_rate", "mean")).reset_index())
    return s, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/metrics.csv")
    ap.add_argument("--curve-attack", default=None)
    ap.add_argument("--curve-budget", type=float, default=None)
    ap.add_argument("--curve-timing", default=None)
    a = ap.parse_args()

    m = pd.read_csv(a.csv)
    os.makedirs("results", exist_ok=True)
    s, d = summarize(m)
    out = s.merge(d, on=KEYS, how="left")
    out.to_csv("results/summary.csv", index=False)
    print(out[out["metric"] == "recall"].round(3).to_string(index=False))

    # 1) curves: clean vs attacked vs defended (recall per window, mean over seeds)
    attacked = m[m["attack"] != "none"]
    if a.curve_attack is None:           # default: the attack with the largest recall drop
        rs = s[s["metric"] == "recall"].sort_values("ASR", ascending=False)
        pick = rs.iloc[0] if len(rs) else None
    else:
        pick = pd.Series(dict(attack=a.curve_attack, budget=a.curve_budget, timing=a.curve_timing))
    if pick is not None:
        sel = attacked[(attacked["attack"] == pick["attack"]) & np.isclose(attacked["budget"], pick["budget"])
                       & (attacked["timing"] == pick["timing"])]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, col in zip(axes, ("recall", "pr_auc")):
            for lab, df, style in (("No attack", m[(m["attack"] == "none") & (m["defense"] == "off")], "k-"),
                                   ("Attacked", sel[sel["defense"] == "off"], "r--"),
                                   ("Attacked + defense", sel[sel["defense"] == "on"], "g-")):
                if df.empty:
                    continue
                g = df.groupby("window")[col].agg(["mean", "std"]).fillna(0)
                ax.plot(g.index, g["mean"], style, label=lab)
                ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.12, color=style[0])
            ax.set_xlabel("window"); ax.set_ylabel(col + " (clean test set)"); ax.grid(alpha=.3)
        axes[0].legend()
        fig.suptitle(f"{pick['attack']}, budget {pick['budget']}, {pick['timing']} (mean ± std over seeds)")
        fig.tight_layout(); fig.savefig("results/fig_curves.png", dpi=150); plt.close(fig)

    # 2) ASR and recovery bars
    r = s[s["metric"] == "recall"].copy()
    if not r.empty:
        r["label"] = r["attack"] + "\n" + r["budget"].astype(str) + " " + r["timing"]
        fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(r)), 4.2))
        x = np.arange(len(r)); w = 0.38
        ax.bar(x - w / 2, r["ASR"], w, label="Attack success rate (recall drop)", color="#c0392b")
        ax.bar(x + w / 2, r["recovery"], w, label="Defense recovery", color="#27ae60")
        ax.set_xticks(x); ax.set_xticklabels(r["label"], fontsize=7); ax.axhline(0, color="k", lw=.6)
        ax.set_ylabel("fraction"); ax.legend(); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig("results/fig_asr_recovery.png", dpi=150); plt.close(fig)

    # 3) detection quality
    if not d.empty:
        d = d.copy(); d["label"] = d["attack"] + "\n" + d["budget"].astype(str) + " " + d["timing"]
        fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(d)), 4.2))
        x = np.arange(len(d)); w = 0.27
        for i, (c, col) in enumerate((("det_precision", "#2980b9"), ("det_recall", "#8e44ad"), ("det_fpr", "#e67e22"))):
            ax.bar(x + (i - 1) * w, d[c].fillna(0), w, label=c, color=col)
        ax.set_xticks(x); ax.set_xticklabels(d["label"], fontsize=7); ax.legend(); ax.grid(axis="y", alpha=.3)
        fig.tight_layout(); fig.savefig("results/fig_detection.png", dpi=150); plt.close(fig)

    print("\nSaved: results/summary.csv, fig_curves.png, fig_asr_recovery.png, fig_detection.png")


if __name__ == "__main__":
    main()
