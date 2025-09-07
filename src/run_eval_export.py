#!/usr/bin/env python3
import argparse
import json
import math
import os

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    csv = os.path.join(a.run, "results.csv")
    df = pd.read_csv(csv)
    last = df.iloc[-1].to_dict()

    def pick(keys):
        for k in keys:
            if k in last and last[k] is not None:
                v = last[k]
                if isinstance(v, float) and math.isnan(v):
                    continue
                try:
                    return float(v)
                except Exception:
                    pass
        return None

    # Prefer (B) metrics if present, otherwise fallback
    summary = {
        "mAP50": pick(["metrics/mAP50(B)", "metrics/mAP50"]),
        "mAP50-95": pick(["metrics/mAP50-95(B)", "metrics/mAP50-95"]),
        "precision": pick(["metrics/precision(B)", "metrics/precision"]),
        "recall": pick(["metrics/recall(B)", "metrics/recall"]),
    }

    with open(os.path.join(a.out, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    def find_col(cols):
        for c in cols:
            if c in df.columns:
                return c
        return None

    rec_col = find_col(["metrics/recall(B)", "metrics/recall"])
    prec_col = find_col(["metrics/precision(B)", "metrics/precision"])

    plt.figure()
    if rec_col and prec_col:
        plt.plot(df[rec_col], df[prec_col], linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR (training trace)")
    plt.grid(True)
    plt.savefig(os.path.join(a.out, "pr_curve.png"), dpi=200, bbox_inches="tight")

    print("Wrote", a.out, summary)


if __name__ == "__main__":
    main()

