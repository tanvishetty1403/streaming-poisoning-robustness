"""Kafka version of the offline loop.

Producer replays the (attacked) stream into a fresh 3-partition topic, key = nameOrig,
so one account always lands on one partition. Each record carries its window id.
After each window the producer sends a window-end marker to every partition; the
consumer closes window k only after seeing the marker from all partitions, then does
exactly what the offline loop does. --check compares the result with the offline run.
"""
import argparse
import json
import threading
import uuid

import numpy as np
import pandas as pd
import yaml
from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from src.pipeline.data import load_dataset, split_stream_test, featurize
from src.pipeline.learner import OnlineLearner
from src.pipeline.metrics import eval_clean, eval_detection
from src.pipeline.replay import replay_windows
from src.pipeline.run_offline import (COLUMNS, append_csv, get_attack, get_defense,
                                      run_offline)

PARTS, BROKER = 3, "localhost:9092"


def make_topic(topic):
    admin = AdminClient({"bootstrap.servers": BROKER})
    for f in admin.create_topics([NewTopic(topic, num_partitions=PARTS, replication_factor=1)]).values():
        f.result()


def produce(topic, poisoned, window_steps):
    p = Producer({"bootstrap.servers": BROKER, "linger.ms": 20, "enable.idempotence": True})
    for k, win in replay_windows(poisoned, window_steps):
        for rec in win.to_dict("records"):
            rec["_w"] = k
            key = str(rec.get("nameOrig", rec["record_id"]))
            p.produce(topic, key=key, value=json.dumps(rec))
            p.poll(0)
        for part in range(PARTS):
            p.produce(topic, partition=part, value=json.dumps({"_end": k}))
        p.flush()


def consume(topic, dataset, test, cfg, defense, meta, wids, seed, out):
    lcfg = cfg["learner"]
    c = Consumer({"bootstrap.servers": BROKER, "group.id": f"g-{uuid.uuid4().hex}",
                  "auto.offset.reset": "earliest"})
    c.subscribe([topic])
    X_test, y_test = featurize(test, dataset), test["isFraud"].to_numpy()
    learner, defend, state = OnlineLearner(lcfg["fraud_weight"], seed), get_defense(defense), {}
    bufs, ends, rows, n_seen, i = {}, {}, [], 0, 0
    while i < len(wids):
        m = c.poll(1.0)
        if m is None or m.error():
            continue
        rec = json.loads(m.value())
        if "_end" in rec:
            ends.setdefault(rec["_end"], set()).add(m.partition())
        else:
            bufs.setdefault(rec["_w"], []).append(rec)
        while i < len(wids) and len(ends.get(wids[i], ())) == PARTS:   # window complete on all partitions
            k = wids[i]
            win = pd.DataFrame(bufs.pop(k, [])).drop(columns="_w").sort_values("record_id")
            visible = win.drop(columns=["is_poisoned"])
            accepted, quarantined, state = defend(visible, state)
            acc = win[win["record_id"].isin(accepted["record_id"])]
            learner.partial_fit(featurize(acc, dataset), acc["isFraud"].to_numpy())
            n_seen += len(win)
            row = dict(meta, run_id="kafka", window=k, step_end=int(win["step"].max()),
                       n_seen=n_seen, n_quarantined=len(quarantined))
            row.update(eval_clean(learner, X_test, y_test, lcfg["threshold"]))
            row.update(eval_detection(win, quarantined) if defense != "off"
                       else {"det_precision": np.nan, "det_recall": np.nan})
            rows.append(row)
            i += 1
    c.close()
    out["df"] = pd.DataFrame(rows, columns=COLUMNS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="paysim")
    ap.add_argument("--attack", default="none")
    ap.add_argument("--budget", type=float, default=0.0)
    ap.add_argument("--timing", default="uniform")
    ap.add_argument("--defense", default="off")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="config/default.yaml")
    ap.add_argument("--check", action="store_true", help="also run offline and compare")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dcfg = cfg["datasets"][a.dataset]

    df = load_dataset(a.dataset, dcfg, a.seed)
    stream, test = split_stream_test(df, dcfg["test_from_step"])
    poisoned, mask = get_attack(a.attack)(stream, budget=a.budget, timing=a.timing, seed=a.seed)
    poisoned = poisoned.copy()
    poisoned["is_poisoned"] = mask
    wids = sorted(set(((poisoned["step"] - poisoned["step"].min()) // dcfg["window_steps"]).tolist()))

    topic = f"transactions-{uuid.uuid4().hex[:8]}"
    make_topic(topic)
    meta = {"dataset": a.dataset, "attack": a.attack, "budget": a.budget,
            "timing": a.timing, "defense": a.defense, "seed": a.seed}
    out = {}
    t = threading.Thread(target=consume, args=(topic, a.dataset, test, cfg, a.defense,
                                               meta, wids, a.seed, out))
    t.start()
    produce(topic, poisoned, dcfg["window_steps"])
    t.join()

    kdf = out["df"]
    append_csv(kdf, cfg["results_csv"])
    print(kdf.tail(3).to_string(index=False))

    if a.check:
        odf = run_offline(a.dataset, a.attack, a.budget, a.timing, a.defense, a.seed, cfg, data=df)
        cols = ["window", "n_seen", "n_quarantined", "pr_auc", "recall", "f1"]
        ok = np.allclose(kdf[cols].to_numpy(float), odf[cols].to_numpy(float), equal_nan=True, atol=1e-9)
        print("KAFKA MATCHES OFFLINE" if ok else "MISMATCH between Kafka and offline runs")


if __name__ == "__main__":
    main()