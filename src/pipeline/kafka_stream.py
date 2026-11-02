"""Kafka version of the offline loop.

Producer replays the (attacked) train stream into a fresh 3-partition topic.
Key = nameOrig for PaySim, record_id for datasets without accounts (ULB), so records
spread over partitions either way. After each event-time window the producer sends a
window-end marker to every partition; the consumer closes window k only after it has
seen the marker from all partitions, then runs the SAME StreamRun code as the offline
loop (src/pipeline/core.py). --check runs the offline loop on the same data and compares.
"""
import argparse
import json
import threading
import time
import uuid

import numpy as np
import yaml
from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient, NewTopic

from src.pipeline.core import StreamRun, prepare_run, records_to_window
from src.pipeline.data import load_dataset
from src.attacks.sweep import load_clean
from src.pipeline.replay import replay_windows
from src.pipeline.run_offline import append_csv, run_offline

PARTS, BROKER, IDLE_TIMEOUT = 3, "localhost:9092", 120
CHECK_COLS = ["window", "n_seen", "n_quarantined", "pr_auc", "recall", "f1", "det_precision",
              "det_recall", "det_fpr", "time_to_detect", "backdoor_hit_rate"]


def partition_key(rec):
    """PaySim has accounts (nameOrig); ULB does not, so fall back to record_id."""
    return str(rec["nameOrig"]) if "nameOrig" in rec else str(rec["record_id"])


def _native(o):                       # json fallback for numpy scalars
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o))


def make_topic(topic):
    admin = AdminClient({"bootstrap.servers": BROKER})
    for f in admin.create_topics([NewTopic(topic, num_partitions=PARTS, replication_factor=1)]).values():
        f.result()


def produce(topic, poisoned, window_steps):
    p = Producer({"bootstrap.servers": BROKER, "linger.ms": 20, "enable.idempotence": True})

    def send(value, key=None, partition=-1):
        while True:
            try:
                p.produce(topic, key=key, value=value, partition=partition)
                p.poll(0)
                return
            except BufferError:          # local queue full: wait for deliveries, retry
                p.poll(0.5)

    for k, win in replay_windows(poisoned, window_steps):
        for rec in win.to_dict("records"):
            rec["_w"] = k
            send(json.dumps(rec, default=_native), key=partition_key(rec))
        for part in range(PARTS):
            send(json.dumps({"_end": k}), partition=part)
        p.flush()


def consume(topic, run, wids, dtypes, out):
    c = None
    try:
        c = Consumer({"bootstrap.servers": BROKER, "group.id": f"g-{uuid.uuid4().hex}",
                      "auto.offset.reset": "earliest"})
        c.subscribe([topic])
        bufs, ends, i, last = {}, {}, 0, time.time()
        while i < len(wids):
            m = c.poll(1.0)
            if m is None or m.error():
                if time.time() - last > IDLE_TIMEOUT:
                    raise TimeoutError(f"no Kafka messages for {IDLE_TIMEOUT}s")
                continue
            last = time.time()
            rec = json.loads(m.value())
            if "_end" in rec:
                ends.setdefault(rec["_end"], set()).add(m.partition())
            else:
                bufs.setdefault(rec["_w"], []).append(rec)
            while i < len(wids) and len(ends.get(wids[i], ())) == PARTS:   # window complete everywhere
                k = wids[i]
                run.step(k, records_to_window(bufs.pop(k), dtypes))
                i += 1
        out["df"] = run.result()
    except Exception as e:               # surface thread errors to main()
        out["error"] = e
    finally:
        if c is not None:
            c.close()


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
    ap.add_argument("--clean-dir", help="use <dir>/stream_clean.csv.gz + test_clean.csv.gz instead of data/raw (no raw PaySim needed)")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    dcfg = cfg["datasets"][a.dataset]

    df = load_clean(a.clean_dir) if a.clean_dir else load_dataset(a.dataset, dcfg, a.seed)
    # same prepare_run as the offline loop: apply_attack gets the identical train stream
    poisoned, test, dcfg = prepare_run(a.dataset, a.attack, a.budget, a.timing, a.seed, cfg, data=df)
    wids = sorted(set(((poisoned["step"] - poisoned["step"].min()) // dcfg["window_steps"]).tolist()))
    run = StreamRun(a.dataset, test, cfg, a.attack, a.budget, a.timing, a.defense, a.seed, "kafka")

    topic = f"transactions-{uuid.uuid4().hex[:8]}"        # fresh topic per run
    make_topic(topic)
    out = {}
    t = threading.Thread(target=consume, args=(topic, run, wids, poisoned.dtypes.to_dict(), out))
    t.start()
    produce(topic, poisoned, dcfg["window_steps"])
    t.join()
    if "error" in out:
        raise out["error"]

    kdf = out["df"]
    append_csv(kdf, cfg["results_csv"])
    print(kdf.tail(3).to_string(index=False))

    if a.check:
        odf = run_offline(a.dataset, a.attack, a.budget, a.timing, a.defense, a.seed, cfg, data=df)
        ok = np.allclose(kdf[CHECK_COLS].to_numpy(float), odf[CHECK_COLS].to_numpy(float),
                         equal_nan=True, atol=1e-9)
        print("KAFKA MATCHES OFFLINE" if ok else "MISMATCH between Kafka and offline runs")
        if not ok:
            diff = (kdf[CHECK_COLS].astype(float) - odf[CHECK_COLS].astype(float)).abs()
            print("columns that differ:", [c for c in CHECK_COLS if (diff[c] > 1e-9).any()])


if __name__ == "__main__":
    main()