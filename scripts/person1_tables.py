"""Person 1: build clean vs attacked vs defended tables from results/summary/*.csv.
Run from the repo root:  python scripts\\person1_tables.py
ASR here = (clean_PRAUC - attacked_PRAUC) / clean_PRAUC  (undefended), per the project guide.
Verify against the repo's own ASR code before using it in the report.
"""
import pandas as pd

S = "results/summary/"
u = pd.read_csv(S + "utility.csv")
d = pd.read_csv(S + "detection.csv")

print("attack values:", u.attack.unique().tolist())
print("budgets:", u.budget.unique().tolist(), "seeds:", sorted(u.seed.unique().tolist()))

clean_names = {"none", "clean", "no_attack", "baseline"}
is_clean = u.attack.str.lower().isin(clean_names)
clean = u[is_clean].groupby("defense")["pr_auc"].agg(["mean", "std"])
print("\nCLEAN BASELINE (PR-AUC, mean/std over seeds):\n", clean.round(4))
c_off = clean.loc["off", "mean"]

a = u[~is_clean]
g = a.groupby(["attack", "timing", "budget", "defense"])["pr_auc"].agg(["mean", "std"]).unstack("defense")
g.columns = [f"{x}_{y}" for x, y in g.columns]
g = g.reset_index()
g["clean_prauc"] = c_off
g["ASR"] = (c_off - g["mean_off"]) / c_off
den = c_off - g["mean_off"]
g["recovery"] = ((g["mean_on"] - g["mean_off"]) / den).where(den > 0.01)  # blank if no real damage

bd = a[a.attack == "backdoor"].groupby(["timing", "budget", "defense"])["backdoor_hit_rate"].mean().unstack("defense")
bd.columns = [f"bd_hit_{c}" for c in bd.columns]

det = d.groupby(["attack", "timing", "budget"])[["precision", "recall", "clean_loss", "time_to_detect"]].agg(["mean", "std"])
det.columns = [f"det_{x}_{y}" for x, y in det.columns]
det = det.reset_index()

out = g.merge(det, on=["attack", "timing", "budget"], how="left")
out.to_csv(S + "person1_final_table.csv", index=False)
bd.reset_index().to_csv(S + "person1_backdoor_hit.csv", index=False)

pd.set_option("display.width", 250)
cols = ["attack", "timing", "budget", "clean_prauc", "mean_off", "mean_on", "ASR", "recovery",
        "det_precision_mean", "det_recall_mean"]
print("\nMAIN TABLE:\n", out[cols].round(3).to_string(index=False))
print("\nBACKDOOR HIT RATE (mean over seeds):\n", bd.round(3).to_string())
print("\nSaved: person1_final_table.csv, person1_backdoor_hit.csv in results/summary/")
