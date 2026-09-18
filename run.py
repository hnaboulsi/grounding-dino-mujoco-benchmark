"""Run the complete controlled couch experiment once."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from detector import GroundingDINO
from evaluate import evaluate
from render import render_scene
from scene import SCENE

OUTPUT_DIR = Path("output")
DEVICE = "cpu"


def main() -> None:
    rendered = render_scene(SCENE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rendered.rgb, mode="RGB").save(OUTPUT_DIR / "rgb.png")
    Image.fromarray(rendered.target_mask, mode="L").save(OUTPUT_DIR / "target_mask.png")

    detector = GroundingDINO(device=DEVICE)
    detections = detector.detect(rendered.rgb, SCENE.prompt)
    result = {
        "scene": SCENE.to_dict(),
        "ground_truth": rendered.ground_truth,
        "detections": [asdict(detection) for detection in detections],
        "detector": detector.metadata(),
        "evaluation": evaluate(detections, rendered.ground_truth),
    }
    (OUTPUT_DIR / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved RGB, target mask, and result JSON under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
