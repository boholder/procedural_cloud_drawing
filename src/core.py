from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.cg_hlsl import (
    dot,
    lerp,
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
    color: tuple[float, float, float] = (0.1, 0.1, 0.15)
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
    day_sky_color: tuple[float, float, float] = (0.53, 0.81, 0.92)
    night_color: tuple[float, float, float] = (0.1, 0.1, 0.2)
    ground_color: tuple[float, float, float] = (0.6, 0.6, 1.0)
    ground_offset: float = 0.0
    ground_reflection_strength: float = 0.2
    ground_reflection_falloff: float = 10.0
    sun_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    enable_sun: bool = True
    sun_color: tuple[float, float, float] = (1.0, 0.95, 0.8)
    sun_size: float = 0.1
    sun_intensity: float = 3.0
    sun_cloud_multiplier: float = 0.1
    enable_moon: bool = False
    moon_color: tuple[float, float, float] = (0.7, 0.7, 0.85)
    moon_size: float = 0.1
    moon_intensity: float = 2.0
    moon_cloud_multiplier: float = 0.1
    background_clouds: CloudSettings = field(default_factory=CloudSettings)


def day_night_blend(sun_direction: np.ndarray) -> float:
    return float(saturate(sun_direction[1] * 2.0 + 1.0))


def get_base_sky_and_ground_color(tex_coord: np.ndarray, settings: SkySettings, blend: float) -> np.ndarray:
    color = lerp(np_arr(settings.night_color), np_arr(settings.day_sky_color), blend)
    band = saturate(np.cos((tex_coord[..., 1] + settings.ground_offset) * np.pi))
    reflection = band[..., None] ** settings.ground_reflection_falloff
    return color + reflection * settings.ground_reflection_strength * np_arr(settings.ground_color)


def get_sun_color(sky_dir: np.ndarray, sun_dir: np.ndarray, settings: SkySettings) -> np.ndarray:
    if not settings.enable_sun:
        return np.zeros_like(sky_dir)
    angle = np.arccos(np.clip(dot(sky_dir, sun_dir), -1.0, 1.0))
    disc = smoothstep(settings.sun_size, 0.0, angle)
    return np_arr(settings.sun_color) * settings.sun_intensity * disc[..., None]


def composite_moon(color: np.ndarray, sky_dir: np.ndarray, moon_dir: np.ndarray, settings: SkySettings) -> np.ndarray:
    """Analytic moon disc/glow; texture sampling is intentionally omitted."""
    if not settings.enable_moon:
        return color
    angle = np.arccos(np.clip(dot(sky_dir, moon_dir), -1.0, 1.0))
    disc = (angle <= settings.moon_size).astype(float)
    radial = angle / max(settings.moon_size * 2.0, 1e-12)
    glow = smoothstep(0.7, 0.5, radial) ** 2
    result = color + np_arr(settings.moon_color) * (0.35 * settings.moon_intensity * glow[..., None])
    moon = np_arr(settings.moon_color) * settings.moon_intensity
    return lerp(result, moon, disc[..., None])


def composite_clouds(
    color: np.ndarray,
    tex_coord: np.ndarray,
    sky_dir: np.ndarray,
    sun_dir: np.ndarray,
    moon_dir: np.ndarray,
    blend: float,
    clouds: CloudSettings,
    settings: SkySettings,
    time: float,
) -> np.ndarray:
    if not clouds.enabled:
        return color
    # Unity _Time.x is t/20; this deliberately preserves the shader's speed.
    cloud_uv = rotate_vector_around_axis(tex_coord, time / 20.0 * clouds.speed, (0.1, 0.2, 0.3))
    noise = fractal_noise(
        cloud_uv * clouds.scale,
        clouds.noise_frequency,
        clouds.noise_amplitude,
        clouds.octaves,
        clouds.warp_frequency,
        clouds.algorithm,
    )
    contrasted = np.power(np.maximum(noise, 0.0), clouds.density)
    smooth_range = clouds.smoothness * 0.5
    coverage = float(saturate(1.0 - clouds.coverage))
    amount = smoothstep(coverage - smooth_range, coverage + smooth_range, contrasted)

    cloud_color = np_arr(clouds.color)
    tint = lerp(cloud_color * 0.05 + (0.025, 0.05, 0.1), cloud_color, blend)
    blended = np.broadcast_to(tint, color.shape).copy()
    sun_rim_color = np.zeros_like(color)
    moon_rim_color = np.zeros_like(color)
    if settings.enable_sun:
        rim = saturate(dot(sky_dir, sun_dir)) ** 8.0
        sun_rim_color = (
            np_arr(settings.sun_color) * (rim * settings.sun_intensity * settings.sun_cloud_multiplier)[..., None]
        )
        blended += lerp(moon_rim_color, sun_rim_color, blend) * 2.0
    if settings.enable_moon:
        rim = saturate(dot(sky_dir, moon_dir)) ** 32.0
        moon_rim_color = (
            np_arr(settings.moon_color) * (rim * settings.moon_intensity * settings.moon_cloud_multiplier)[..., None]
        )
        blended += lerp(moon_rim_color, sun_rim_color, blend) * 2.0
    lower_sky_rim = 1.0 - saturate(dot(sky_dir, (0.0, -1.0, 0.0)))
    blended += blended * lower_sky_rim[..., None] ** 2

    return lerp(color, blended, amount[..., None])


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
    settings: SkySettings | None = None,
) -> np.ndarray:
    """Render the HDR RGB view of a camera located at world origin."""
    settings = settings or SkySettings()
    sky_dir = camera_view_directions(np_arr(camera_direction), width, height)
    tex_coord = sky_dir
    sun_dir = normalize(settings.sun_direction)
    moon_dir = -sun_dir
    blend = day_night_blend(sun_dir)
    color = get_base_sky_and_ground_color(tex_coord, settings, blend)
    # color = composite_moon(color, sky_dir, moon_dir, settings)
    # color += get_sun_color(sky_dir, sun_dir, settings)
    color = composite_clouds(
        color,
        tex_coord,
        sky_dir,
        sun_dir,
        moon_dir,
        blend,
        settings.background_clouds,
        settings,
        time,
    )
    return color


def save_sky(path: str | Path, image: np.ndarray) -> None:
    # Reinhard tone mapping retains the shader's HDR highlights in an 8-bit PNG.
    mapped = np.maximum(image, 0.0) / (1.0 + np.maximum(image, 0.0))
    mapped = np.power(mapped, 1.0 / 2.2)
    Image.fromarray(np.uint8(np.clip(mapped, 0.0, 1.0) * 255.0), "RGB").save(path)
