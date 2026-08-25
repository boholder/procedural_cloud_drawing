from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from src.cg_hlsl import (
    dot,
    normalize,
    saturate,
    smoothstep,
)
from src.noise import NoiseAlgorithm, fractal_noise, rotate_vector_around_axis


def np_arr(value: tuple[float, float, float]) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


@dataclass
class CloudSettings:
    enabled: bool = True
    algorithm: NoiseAlgorithm = "simplex"
    coverage: float = 0.5
    density: float = 1.0
    scale: float = 1.5
    speed: float = 0.2
    smoothness: float = 0.2
    octaves: int = 4
    noise_frequency: float = 1.0
    noise_amplitude: float = 1.0
    warp_frequency: float = 0.75


@dataclass
class SkySettings:
    sky_color: tuple[float, float, float] = (0.53, 0.81, 0.92)


def get_base_sky_color(shape: tuple[int, ...], settings: SkySettings) -> np.ndarray:
    return np.broadcast_to(np_arr(settings.sky_color), (*shape, 3)).copy()


def composite_clouds(
    tex_coord: np.ndarray,
    settings: CloudSettings,
    time: float,
) -> np.ndarray:
    # Unity _Time.x is t/20; this deliberately preserves the shader's speed.
    cloud_uv = rotate_vector_around_axis(tex_coord, time / 20.0 * settings.speed, (0.1, 0.2, 0.3))
    noise = fractal_noise(
        cloud_uv * settings.scale,
        settings.noise_frequency,
        settings.noise_amplitude,
        settings.octaves,
        settings.warp_frequency,
        settings.algorithm,
    )
    contrasted = np.power(np.maximum(noise, 0.0), settings.density)
    smooth_range = settings.smoothness * 0.5
    coverage = float(saturate(1.0 - settings.coverage))
    amount = smoothstep(coverage - smooth_range, coverage + smooth_range, contrasted)
    return amount[..., None]


def camera_view_directions(camera_direction: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return world-space perspective rays from the origin through each pixel.

    ``camera_direction`` is the center of the view.  The virtual camera uses
    world +Y as up and a fixed 90-degree vertical field of view.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")

    forward = normalize(camera_direction)

    world_up = np.array((0.0, 1.0, 0.0), dtype=np.float32)
    if abs(float(dot(forward, world_up))) > 0.999:
        world_up = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    right = normalize(np.cross(world_up, forward))
    camera_up = normalize(np.cross(forward, right))

    aspect_ratio = width / height
    screen_x = ((np.arange(width) + 0.5) / width * 2.0 - 1.0) * aspect_ratio
    screen_y = 1.0 - (np.arange(height) + 0.5) / height * 2.0
    x, y = np.meshgrid(screen_x, screen_y)
    rays = forward + x[..., None] * right + y[..., None] * camera_up
    return normalize(rays)


def render_sky(
    camera_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    width: int = 800,
    height: int = 400,
    time: float = 0.0,
    sky_settings: SkySettings | None = None,
    cloud_settings: CloudSettings | None = None,
) -> np.ndarray:
    """Render the HDR RGB view of a camera located at world origin."""
    sky_settings = sky_settings or SkySettings()
    sky_dir = camera_view_directions(np_arr(camera_direction), width, height)
    tex_coord = sky_dir
    base_color = get_base_sky_color(sky_dir.shape[:-1], sky_settings)
    mix_color = base_color
    if cloud_settings.enabled:
        cloud_color = composite_clouds(tex_coord, cloud_settings, time)
        mix_color += cloud_color
    return mix_color


def save_sky(path: str | Path, image: np.ndarray) -> None:
    # Reinhard tone mapping retains the shader's HDR highlights in an 8-bit PNG.
    mapped = np.maximum(image, 0.0) / (1.0 + np.maximum(image, 0.0))
    mapped = np.power(mapped, 1.0 / 2.2)
    Image.fromarray(np.uint8(np.clip(mapped, 0.0, 1.0) * 255.0), "RGB").save(path)
