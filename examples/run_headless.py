# -*- coding: utf-8 -*-
"""Example of using the model without the graphical interface.

Runs a simulation, prints a summary and saves a CSV + figure. Shows how to
call model_runner from your own script (batch runs, sensitivity studies...).

    python examples/run_headless.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from model_runner import RunConfig, run


def main():
    cfg = RunConfig(
        month=2, n_households=50, interval_seconds=600,
        period_type=1, iterations=5,
        reactive=True, localize=True, efficiency=False, pv=False,
    )
    res = run(cfg, progress=lambda p, m: print(f"  [{p:3d}%] {m}"))

    N = res["meta"]["n_households"]
    total = res["total"]
    print("\n=== Summary ===")
    print(f"Mean active power:      {total.mean() / N:6.1f} W/CP")
    print(f"Peak:                  {total.max() / N:6.1f} W/CP")
    print(f"Seasonal factor (TDD): {res['meta']['localize_factor']:.3f}")
    if res["reactive"] is not None:
        print(f"Reactive power Q0:     {res['reactive'][0] / N:6.1f} var/CP (constant, capacitive)")
    print("Appliance groups (mean W/CP):")
    for g, arr in res["groups"].items():
        print(f"   {g:20s} {arr.mean() / N:6.1f}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.DataFrame({"hour": res["t"], "P_W": total})
    for g, arr in res["groups"].items():
        df[f"P_{g}_W"] = arr
    if res["reactive"] is not None:
        df["Q_var"] = res["reactive"]
    csv_path = os.path.join(out_dir, "output.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        bottom = np.zeros_like(total)
        for g, arr in res["groups"].items():
            ax.fill_between(res["t"], bottom, bottom + arr / N, label=g, step="mid")
            bottom = bottom + arr / N
        ax.plot(res["t"], total / N, "k-", lw=1.2, label="Total")
        ax.set_xlabel("Hour")
        ax.set_ylabel("Power (W/CP)")
        ax.set_xlim(0, 24)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        png_path = os.path.join(out_dir, "output.png")
        fig.savefig(png_path, dpi=110)
        print(f"Figure saved: {png_path}")
    except Exception as e:
        print(f"(figure skipped: {e})")


if __name__ == "__main__":
    main()
