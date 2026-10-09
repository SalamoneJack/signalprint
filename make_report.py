"""Collect runs/*/result.json -> figures/ + results table (results.md). Model selection uses VALIDATION accuracy only."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # fixed categorical order, one per experiment
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUES = LinearSegmentedColormap.from_list("blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})


def main():
    os.makedirs(FIG, exist_ok=True)
    res = []
    for p in sorted(glob.glob(os.path.join(HERE, "runs", "*", "result.json"))):
        with open(p) as f:
            res.append(json.load(f))
    if not res:
        print("no results yet")
        return
    color = {r["cfg"]["name"]: SERIES[i % len(SERIES)] for i, r in enumerate(res)}

    # ---- training curves: validation window accuracy per epoch
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    for r in res:
        n, h = r["cfg"]["name"], r["history"]
        ep = [e["epoch"] for e in h]
        va = [100 * e["val_win_acc"] for e in h]
        ax.plot(ep, va, color=color[n], lw=2, label=n)
        ax.plot([r["best_epoch"]], [100 * r["val"]["win_acc"]], "o", ms=8, color=color[n], mec=SURF, mew=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy, per window (%)")
    ax.set_title("Validation accuracy by epoch (dot = best checkpoint)", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "val_curves.png"), dpi=150)
    plt.close(fig)

    # ---- accuracy comparison: same-day held-out recordings vs unseen day
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [r["cfg"]["name"] for r in res]
    x = np.arange(len(res))
    for j, (split, lab, alpha) in enumerate([("test", "Test: held-out recordings, days 1-4", 1.0),
                                             ("test_crossday", "Test: unseen day 5", 0.45)]):
        vals = [100 * r[split]["win_acc"] for r in res]
        bars = ax.bar(x + (j - 0.5) * 0.38, vals, 0.36, color=[color[n] for n in names], alpha=alpha,
                      edgecolor=SURF, linewidth=2, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, "%.1f" % v, ha="center", color=INK2, fontsize=8)
    ax.axhline(4, color=INK2, lw=1, ls="--")
    ax.text(len(res) - 0.5, 5, "chance 4%", color=INK2, ha="right", fontsize=8)
    ax.set_xticks(x, names, fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("per-window accuracy (%)")
    ax.set_title("Test accuracy: solid = held-out recordings, faded = unseen day", loc="left", color=INK)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "test_accuracy.png"), dpi=150)
    plt.close(fig)

    # ---- confusion matrices for the model chosen on validation
    best = max(res, key=lambda r: r["val"]["win_acc"])
    for split in ("test", "test_crossday"):
        d = np.load(os.path.join(HERE, "runs", best["cfg"]["name"], "preds_%s.npz" % split))
        cm = confusion_matrix(d["y"], d["pred"], labels=range(25), normalize="true")
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(100 * cm, cmap=BLUES, vmin=0, vmax=100)
        ax.set_xticks(range(0, 25, 2), [str(i + 1) for i in range(0, 25, 2)])
        ax.set_yticks(range(0, 25, 2), [str(i + 1) for i in range(0, 25, 2)])
        ax.set_xlabel("predicted device")
        ax.set_ylabel("true device")
        ax.grid(False)
        ax.set_title("%s - %s (row %%)" % (best["cfg"]["name"], split), loc="left", color=INK, fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, label="% of true device's windows")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, "confusion_%s.png" % split), dpi=150)
        plt.close(fig)

    # ---- table
    lines = ["| Run | Params | Epochs (best) | Val win | Test win | Test rec | Day-5 win | Day-5 rec |",
             "|---|---|---|---|---|---|---|---|"]
    for r in res:
        lines.append("| %s%s | %s | %d (%d) | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% |" % (
            r["cfg"]["name"], " **(selected)**" if r is best else "", format(r["params"], ","), r["epochs_run"],
            r["best_epoch"], 100 * r["val"]["win_acc"], 100 * r["test"]["win_acc"], 100 * r["test"]["rec_acc"],
            100 * r["test_crossday"]["win_acc"], 100 * r["test_crossday"]["rec_acc"]))
    with open(os.path.join(HERE, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("selected (by val):", best["cfg"]["name"])


if __name__ == "__main__":
    main()
