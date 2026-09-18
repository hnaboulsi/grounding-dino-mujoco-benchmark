"""Render RGB and segmentation ground truth for the controlled scene."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from scene import COUCH_MTL, COUCH_OBJ, Scene, build_xml


@dataclass(frozen=True)
class RenderedScene:
    rgb: np.ndarray
    target_mask: np.ndarray
    ground_truth: dict[str, Any]


def render_scene(scene: Scene) -> RenderedScene:
    """Render the scene and derive the target box from MuJoCo segmentation."""
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    try:
        import mujoco
    except ImportError as error:
        raise RuntimeError("Install MuJoCo with `pip install -r requirements.txt`") from error

    assets = {
        COUCH_OBJ.name: COUCH_OBJ.read_bytes(),
        COUCH_MTL.name: COUCH_MTL.read_bytes(),
    }
    model = mujoco.MjModel.from_xml_string(build_xml(scene), assets=assets)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=scene.height, width=scene.width) as renderer:
        renderer.update_scene(data, camera="evaluation_camera")
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera="evaluation_camera")
        segmentation = renderer.render().copy()

    target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "target")
    if target_id < 0:
        raise RuntimeError("MuJoCo scene does not contain a geom named 'target'")
    target_type = int(mujoco.mjtObj.mjOBJ_GEOM)
    target_mask = (segmentation[:, :, 0] == target_id) & (
        segmentation[:, :, 1] == target_type
    )
    rows, columns = np.nonzero(target_mask)
    if not len(columns):
        raise RuntimeError("The couch is not visible from the configured camera")
    bbox = [
        float(columns.min()),
        float(rows.min()),
        float(columns.max() + 1),
        float(rows.max() + 1),
    ]
    return RenderedScene(
        rgb=rgb,
        target_mask=(target_mask.astype(np.uint8) * 255),
        ground_truth={
            "label": scene.prompt,
            "bbox": bbox,
            "visible_pixels": int(target_mask.sum()),
        },
    )
