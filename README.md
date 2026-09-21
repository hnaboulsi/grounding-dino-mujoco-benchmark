# Grounding DINO MuJoCo Benchmark

Controlled MuJoCo experiment that renders ground truth and measures Grounding DINO couch detections with IoU, precision, and recall-oriented counts.

## Why I Built It

A detector screenshot is not a benchmark. I wanted a small perception experiment where the scene, camera, segmentation mask, target box, model revision, and matching policy are all recorded and reproducible.

## What It Does

The project builds a primitive MuJoCo room, loads a separately licensed CC0 couch mesh, renders RGB and object segmentation, runs Grounding DINO with a text prompt, and writes the detections and evaluation decision to JSON.

## Key Engineering Work

- Built the controlled MuJoCo scene and camera configuration.
- Converted segmentation output into an independently computed ground-truth box.
- Kept model inference behind a small detector adapter.
- Implemented score-ordered matching at a configurable IoU threshold.
- Recorded scene parameters, detections, model revision, and final counts in JSON.
- Added evaluator tests and CI for the lightweight logic.

## Architecture

```text
scene.py → render.py → detector.py → evaluate.py → run.py
     │          │            │             │
  scene      RGB + mask   DINO boxes   IoU / TP / FP / FN
```

Grounding DINO, MuJoCo, PyTorch, and the couch mesh are dependencies. The repository owns the scene construction, ground-truth extraction, adapter boundary, and evaluation policy. Asset attribution is in [`assets/CC0-LICENSE.txt`](assets/CC0-LICENSE.txt).

## Results

The evaluator tests pass locally and in GitHub Actions. A reference run writes `output/rgb.png`, `output/target_mask.png`, and `output/result.json`; those generated artifacts remain ignored until a specific machine/model run is deliberately committed. The current experiment is one target in one controlled scene, not a claim about detector performance in unconstrained robot perception.

## Tech Stack

Python, MuJoCo, Grounding DINO through Hugging Face Transformers, PyTorch, NumPy, Pillow, pytest, and GitHub Actions.

## Running Locally

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libosmesa6
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pytest -q
MUJOCO_GL=osmesa python run.py
```

The first detector run downloads the pinned Grounding DINO Tiny model from Hugging Face. Set `DEVICE = "cuda"` in `run.py` only when the host has a compatible CUDA setup.

MIT License for the code. The couch asset is provided separately under CC0.
