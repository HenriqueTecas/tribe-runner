"""Minimal TRIBE v2 inference: still images + (optional) text.

Edit IMAGES, TEXT, and durations below, then:
    python run_image_text.py
"""

import os
from pathlib import Path

import pandas as pd

from tribev2 import TribeModel
from tribev2.eventstransforms import CreateVideosFromImages, TextToEvents

CACHE = os.environ.get("TRIBE_CACHE", "/workspace/.cache/tribev2")
WORKDIR = Path(os.environ.get("WORKDIR", "/workspace"))

# (path, duration_seconds) pairs. Duration = how long the image is "shown".
IMAGES: list[tuple[str, float]] = [
    (str(WORKDIR / "io/img1.png"), 2.0),
    (str(WORKDIR / "io/img2.png"), 2.0),
]

# Optional caption read alongside the images. Empty string disables text.
TEXT = ""


def build_image_events(images: list[tuple[str, float]]) -> pd.DataFrame:
    rows, t = [], 0.0
    for path, duration in images:
        rows.append({
            "type": "Image",
            "filepath": path,
            "start": t,
            "duration": duration,
            "timeline": "default",
            "subject": "default",
        })
        t += duration
    return pd.DataFrame(rows), t


def main() -> None:
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE)

    image_df, image_end = build_image_events(IMAGES)
    image_df = CreateVideosFromImages()._run(image_df)  # Image -> short Video clips

    if TEXT.strip():
        text_df = TextToEvents(
            text=TEXT,
            infra={"folder": CACHE, "mode": "retry"},
        ).get_events()
        text_df["start"] = text_df["start"].astype(float) + image_end
        events = pd.concat([image_df, text_df], ignore_index=True)
    else:
        events = image_df

    preds, segments = model.predict(events=events)
    print("preds shape:", preds.shape, "segments:", len(segments))


if __name__ == "__main__":
    main()
