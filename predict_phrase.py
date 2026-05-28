"""Run TRIBE v2 on a phrase and save a brain visualization PNG.

Usage on the pod:
    python predict_phrase.py "your phrase here"
    python predict_phrase.py "your phrase here" --out /workspace/io/brain.png
    # interactive (no arg): you'll be prompted

The output PNG shows predicted fMRI activity averaged across the predicted
segments, on the fsaverage5 cortical mesh, four standard views.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write to file, never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from tribev2 import TribeModel
from tribev2.plotting import PlotBrainNilearn, PlotBrainPyvista
from tribev2.plotting.utils import get_cmap, get_scalar_mappable

pv.OFF_SCREEN = True

CACHE = os.environ.get("TRIBE_CACHE", "/workspace/.cache/tribev2")
WORKDIR = Path(os.environ.get("WORKDIR", "/workspace"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("phrase", nargs="?", help="phrase to predict on (prompts if omitted)")
    p.add_argument(
        "--out",
        default=str(WORKDIR / "io/brain.png"),
        help="output PNG path (default: /workspace/io/brain.png)",
    )
    p.add_argument(
        "--reduction",
        choices=["mean", "max", "first", "last"],
        default="mean",
        help="how to collapse the per-segment predictions to one map",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    phrase = args.phrase or input("Phrase: ").strip()
    if not phrase:
        print("empty phrase, aborting", file=sys.stderr)
        sys.exit(1)

    text_dir = WORKDIR / "io"
    text_dir.mkdir(parents=True, exist_ok=True)
    text_path = text_dir / "phrase.txt"
    text_path.write_text(phrase + "\n", encoding="utf-8")
    print(f"phrase: {phrase!r}")

    print("loading model...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE)

    print("building events...")
    events = model.get_events_dataframe(text_path=str(text_path))

    print("predicting...")
    preds, segments = model.predict(events=events)
    print(f"preds shape: {preds.shape}, segments: {len(segments)}")

    if preds.shape[0] == 0:
        print("no segments kept (phrase too short for any TR window). "
              "try a longer phrase.", file=sys.stderr)
        sys.exit(2)

    reducer = {
        "mean": lambda a: a.mean(axis=0),
        "max": lambda a: a.max(axis=0),
        "first": lambda a: a[0],
        "last": lambda a: a[-1],
    }[args.reduction]
    signal = reducer(preds).astype(np.float32)
    print(f"reduction={args.reduction}, signal range "
          f"[{signal.min():+.3f}, {signal.max():+.3f}]")

    print("rendering brain...")
    pb = PlotBrainNilearn(mesh="fsaverage5", inflate="half", bg_map="sulcal")
    pb.plot_surf(
        signals=signal,
        views=["left", "right", "medial_left", "medial_right"],
        cmap="bwr",
        symmetric_cbar=True,
        colorbar=True,
        colorbar_title="predicted fMRI",
    )
    fig = plt.gcf()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(phrase[:120], fontsize=10)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {out}")
    print(f"\non your laptop:\n  scp root@<pod-ip>:{out} ./brain.png")


if __name__ == "__main__":
    main()
