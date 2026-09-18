"""One editable MuJoCo couch scene."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class Scene:
    width: int = 640
    height: int = 480
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5
    camera_position: Vector3 = (-4.0, 0.0, 1.1)
    camera_look_at: Vector3 = (0.0, 0.0, 0.6)
    couch_position: Vector3 = (0.0, 0.0, 0.0)
    couch_yaw_deg: float = -90.0
    prompt: str = "couch"
    lighting: str = "normal"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


SCENE = Scene()
ASSET_DIR = Path(__file__).parent / "assets"
COUCH_OBJ = ASSET_DIR / "Couch_Large1.obj"
COUCH_MTL = ASSET_DIR / "Couch_Large1.mtl"
COUCH_FLOOR_OFFSET = 0.01
COUCH_SCALE = (0.42, 0.42, 0.42)
COUCH_COLOR = (0.22, 0.38, 0.68, 1.0)


def build_xml(scene: Scene) -> str:
    """Build a primitive room with one independently named couch geom."""
    if scene.lighting not in {"normal", "dim", "bright"}:
        raise ValueError("lighting must be normal, dim, or bright")
    right_x, right_y, up_x, up_y, up_z = _camera_axes(scene)
    ambient, diffuse = {
        "normal": ((0.15, 0.15, 0.15), (0.75, 0.75, 0.75)),
        "dim": ((0.03, 0.03, 0.03), (0.18, 0.18, 0.18)),
        "bright": ((0.30, 0.30, 0.30), (1.0, 1.0, 1.0)),
    }[scene.lighting]
    couch_x, couch_y, couch_z = scene.couch_position
    camera = _numbers(scene.camera_position)
    color = _numbers(COUCH_COLOR)
    scale = _numbers(COUCH_SCALE)
    return f"""<mujoco model="minimal_couch_scene">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>
  <asset>
    <mesh name="target_mesh" file="Couch_Large1.obj" scale="{scale}"/>
  </asset>
  <visual>
    <global offwidth="{scene.width}" offheight="{scene.height}"/>
    <headlight ambient="{_numbers(ambient)}" diffuse="{_numbers(diffuse)}" specular="0.1 0.1 0.1"/>
  </visual>
  <worldbody>
    <geom name="floor" type="box" size="4 4 0.05" pos="0 0 -0.05" rgba="0.58 0.56 0.52 1" contype="0" conaffinity="0"/>
    <geom name="back_wall" type="box" size="4 0.05 1.5" pos="0 4 1.5" rgba="0.86 0.84 0.79 1" contype="0" conaffinity="0"/>
    <geom name="left_wall" type="box" size="0.05 4 1.5" pos="-4 0 1.5" rgba="0.86 0.84 0.79 1" contype="0" conaffinity="0"/>
    <geom name="right_wall" type="box" size="0.05 4 1.5" pos="4 0 1.5" rgba="0.86 0.84 0.79 1" contype="0" conaffinity="0"/>
    <light name="ceiling_light" pos="0 0 2.8" dir="0 0 -1" diffuse="{_numbers(diffuse)}" ambient="0.03 0.03 0.03" castshadow="false"/>
    <camera name="evaluation_camera" pos="{camera}" xyaxes="{right_x:.8g} {right_y:.8g} 0 {up_x:.8g} {up_y:.8g} {up_z:.8g}" resolution="{scene.width} {scene.height}" sensorsize="{scene.width} {scene.height}" focalpixel="{scene.fx:.8g} {scene.fy:.8g}" principalpixel="{(scene.width - 1) / 2 - scene.cx:.8g} {scene.cy - (scene.height - 1) / 2:.8g}"/>
    <body name="target_body" pos="{couch_x:.8g} {couch_y:.8g} {couch_z + COUCH_FLOOR_OFFSET:.8g}" euler="0 0 {scene.couch_yaw_deg:.8g}">
      <geom name="target" type="mesh" mesh="target_mesh" euler="90 0 0" rgba="{color}" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>"""


def _camera_axes(scene: Scene) -> tuple[float, float, float, float, float]:
    forward = tuple(
        look - position for look, position in zip(scene.camera_look_at, scene.camera_position)
    )
    length = math.sqrt(sum(value * value for value in forward))
    if length == 0:
        raise ValueError("camera_position and camera_look_at must differ")
    forward = tuple(value / length for value in forward)
    right = (forward[1], -forward[0], 0.0)
    right_length = math.sqrt(right[0] ** 2 + right[1] ** 2)
    if right_length == 0:
        right = (1.0, 0.0, 0.0)
    else:
        right = (right[0] / right_length, right[1] / right_length, 0.0)
    up = (
        right[1] * forward[2],
        -right[0] * forward[2],
        right[0] * forward[1] - right[1] * forward[0],
    )
    return right[0], right[1], up[0], up[1], up[2]


def _numbers(values: tuple[float, ...]) -> str:
    return " ".join(f"{float(value):.8g}" for value in values)
