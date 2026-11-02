"""Kafka throughput / latency benchmark: partitions x defense on/off -> results/benchmark.csv

Needs Kafka running:  docker compose -f docker/docker-compose.yml up -d
    python scripts/benchmark.py
    python scripts/benchmark.py --partitions 1,2,3 --reps 2

throughput_rps = records processed / wall time (producer + Kafka + consumer + defense + learner + eval)
p95_window_latency_ms = 95th percentile of the per-window processing time in the consumer
Defense overhead is printed as the % throughput lost with the defense on.
"""
import argparse
import os
import platform
import sys
import threading
import time
import uuid

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.pipeline import kafka_stream as ks                                   # noqa: E402
from src.pipeline.core import StreamRun, prepare_run                          # noqa: E402
from src.pipeline.data import load_dataset                                    # noqa: E402


def one_run(df, cfg, dataset, parts, defense, attack, budget, timing, seed):
    ks.PARTS = parts                                  # kafka_stream reads this global at call time
    dcfg = cfg["datasets"][dataset]
    poisoned, test, dcfg = prepare_run(dataset, attack, budget, timing, seed, cfg, data=df)
    wids = sorted(set(((poisoned["step"] - poisoned["step"].min()) // dcfg["window_steps"]).tolist()))
    run = StreamRun(dataset, test, cfg, attack, budget, timing, defense, seed, "bench")

    lat = []
    orig = run.step
    def timed(k, win):
        t = time.perf_counter(); orig(k, win); lat.append((time.perf_counter() - t) * 1000)
    run.step = timed

    topic = f"bench-{uuid.uuid4().hex[:8]}"
    ks.make_topic(topic)
    out = {}
    th = threading.Thread(target=ks.consume, args=(topic, run, wids, poisoned.dtypes.to_dict(), out))
    t0 = time.perf_counter()
    th.start()
    ks.produce(topic, poisoned, dcfg["window_steps"])
    th.join()
    wall = time.perf_counter() - t0
    if "error" in out:
        raise out["error"]
    return dict(dataset=dataset, partitions=parts, defense=defense, records=len(poisoned), windows=len(wids),
                wall_s=wall, throughput_rps=len(poisoned) / wall, p95_window_latency_ms=float(np.percentile(lat, 95)),
                mean_window_latency_ms=float(np.mean(lat)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="paysim")
    ap.add_argument("--partitions", default="1,2,3")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--attack", default="targeted_flip")
    ap.add_argument("--budget", type=float, default=0.5)
    ap.add_argument("--timing", default="burst")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--broker", default="localhost:9092")
    ap.add_argument("--config", default="config/default.yaml")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    ks.BROKER = a.broker
    df = load_dataset(a.dataset, cfg["datasets"][a.dataset], a.seed)

    rows = []
    for rep in range(a.reps):
        for p in [int(x) for x in a.partitions.split(",")]:
            for d in ("off", "on"):
                r = one_run(df, cfg, a.dataset, p, d, a.attack, a.budget, a.timing, a.seed)
                r["rep"] = rep
                rows.append(r)
                print(f"partitions={p} defense={d:3s} {r['throughput_rps']:.0f} rec/s  "
                      f"p95 window latency {r['p95_window_latency_ms']:.0f} ms", flush=True)

    res = pd.DataFrame(rows)
    res["hardware"] = f"{platform.processor() or platform.machine()}, {os.cpu_count()} cores, {platform.system()}"
    os.makedirs("results", exist_ok=True)
    out = cfg["benchmark_csv"]
    res.to_csv(out, mode="a", header=not os.path.exists(out), index=False)

    g = res.groupby(["partitions", "defense"])[["throughput_rps", "p95_window_latency_ms"]].mean().unstack()
    print("\n", g.round(1).to_string())
    for p, sub in res.groupby("partitions"):
        off = sub[sub.defense == "off"].throughput_rps.mean(); on = sub[sub.defense == "on"].throughput_rps.mean()
        print(f"partitions={p}: defense overhead = {(off - on) / off:.1%} throughput")
    print(f"\nrecords per run: {res.records.iloc[0]}, hardware: {res.hardware.iloc[0]}\nSaved to {out}")


if __name__ == "__main__":
    main()
