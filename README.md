# Minimal Grounding DINO MuJoCo experiment

This repository runs one transparent controlled experiment: a primitive MuJoCo room contains one separately loaded CC0 couch OBJ, MuJoCo renders RGB and object segmentation, Grounding DINO predicts a couch box, and the program writes IoU, TP, FP, and FN to JSON.

The execution path is intentionally short:

```text
scene.py → render.py → detector.py → evaluate.py → run.py
```

## Linux setup

Install the headless rendering libraries on Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libosmesa6
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Run the experiment:

```bash
MUJOCO_GL=osmesa python run.py
```

The first detector run downloads the pinned Grounding DINO Tiny model from Hugging Face. The model is CPU-first; to use CUDA, change `DEVICE = "cpu"` to `DEVICE = "cuda"` in `run.py`.

## Output

`output/rgb.png` is the rendered camera image. `output/target_mask.png` is the binary segmentation mask for the couch geom. `output/result.json` contains the scene values, visible-pixel ground truth box, every Grounding DINO detection above confidence `0.35`, model revision, and the final evaluation at IoU `0.50`.

## Edit the experiment

Edit the `SCENE` value in `scene.py`. The fields are deliberately visible in one place:

- `camera_position` and `camera_look_at` control the view.
- `fx`, `fy`, `cx`, and `cy` control the camera intrinsics.
- `couch_position` and `couch_yaw_deg` control target placement.
- `lighting` accepts `normal`, `dim`, or `bright`.
- `prompt` is the Grounding DINO text prompt.

The room shell is built in `build_xml` from floor and wall primitives. The couch remains an independent MuJoCo body named `target`, so segmentation can identify it directly. The OBJ scale, mesh rotation, color, and floor offset are the constants immediately below `SCENE`.

`evaluate.py` contains the only matching policy: score-ordered detections, one target match at IoU `0.50`, and duplicate or unmatched predictions counted as false positives. Change those constants only if the experiment question requires a different operating point.

The asset is CC0. Its source attribution and license are in `assets/CC0-LICENSE.txt`.
