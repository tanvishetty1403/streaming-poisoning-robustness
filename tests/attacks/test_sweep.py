import pandas as pd
import yaml

from src.attacks.sweep import KEY, done_keys, grid, main
from tests.test_harness import fake_df

CFG = yaml.safe_load(open("config/default.yaml"))


def test_grid_adds_one_baseline_per_seed():
    runs = list(grid(["random_flip"], [0.05, 0.2], ["uniform", "burst"], [0, 1]))
    assert sum(r[0] == "none" for r in runs) == 2
    assert len(runs) == 2 * (1 + 4)


def test_sweep_writes_contract_columns_and_resumes(tmp_path, monkeypatch):
    d = fake_df(6000)
    d[d["step"] < 600].to_csv(tmp_path / "stream_clean.csv.gz", index=False)
    d[d["step"] >= 600].to_csv(tmp_path / "test_clean.csv.gz", index=False)
    out = tmp_path / "metrics.csv"
    args = ["--clean-dir", str(tmp_path), "--attacks", "random_flip", "--budgets", "0.2", "--timings", "uniform",
            "--seeds", "0", "--out", str(out)]
    main(args)
    m = pd.read_csv(out)
    assert set(KEY) <= set(m.columns)
    assert set(m["defense"]) == {"off", "on"} and set(m["attack"]) == {"none", "random_flip"}
    n = len(m)
    main(args)                                   # second call: everything already done
    assert len(pd.read_csv(out)) == n
    assert len(done_keys(str(out))) == 4
    pair = m[m["attack"] == "random_flip"].groupby("defense")["n_seen"].max()
    assert pair["off"] == pair["on"]             # same stream replayed for both settings
