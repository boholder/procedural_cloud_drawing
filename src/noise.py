from typing import Callable, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.cg_hlsl import frac, lerp, normalize, saturate, dot

NoiseAlgorithm = Literal["simplex", "opensimplex2s", "perlin", "value", "worley", "curl_domain_warp"]
FloatArray = NDArray[np.floating]


def rotate_vector_around_axis(p: ArrayLike, angle: float, axis: ArrayLike) -> FloatArray:
    p = np.asarray(p, float)
    a = normalize(axis)
    c, s = np.cos(angle), np.sin(angle)
    return p * c + np.cross(a, p) * s + a * dot(p, a)[..., None] * (1.0 - c)


def hash31(p: ArrayLike) -> FloatArray:
    p = np.asarray(p, float)
    return frac(np.sin(np.sum(p * (127.1, 311.7, 74.7), axis=-1)) * 43758.5453)


def hash33(p: ArrayLike) -> FloatArray:
    p = np.asarray(p, float)
    q = np.stack(
        (
            np.sum(p * (127.1, 311.7, 74.7), axis=-1),
            np.sum(p * (269.5, 183.3, 246.1), axis=-1),
            np.sum(p * (113.5, 271.9, 124.6), axis=-1),
        ),
        axis=-1,
    )
    return frac(np.sin(q) * 43758.5453)


def fade(t: ArrayLike) -> FloatArray:
    t = np.asarray(t, float)
    return t**3 * (t * (t * 6.0 - 15.0) + 10.0)


def value3d(p: ArrayLike) -> FloatArray:
    p = np.asarray(p, float)
    i, f = np.floor(p), frac(p)
    w = fade(f)
    values = [hash31(i + (x, y, z)) for z in range(2) for y in range(2) for x in range(2)]
    x00, x10, x01, x11 = (lerp(values[j], values[j + 1], w[..., 0]) for j in (0, 2, 4, 6))
    return lerp(lerp(x00, x10, w[..., 1]), lerp(x01, x11, w[..., 1]), w[..., 2])


def perlin3d(p: ArrayLike) -> FloatArray:
    p = np.asarray(p, float)
    i, f = np.floor(p), frac(p)
    w = fade(f)
    corners = []
    for z in range(2):
        for y in range(2):
            for x in range(2):
                offset = np.array((x, y, z))
                gradient = normalize(hash33(i + offset) * 2.0 - 1.0)
                corners.append(np.sum(gradient * (f - offset), axis=-1))
    x00, x10, x01, x11 = (lerp(corners[j], corners[j + 1], w[..., 0]) for j in (0, 2, 4, 6))
    result = lerp(lerp(x00, x10, w[..., 1]), lerp(x01, x11, w[..., 1]), w[..., 2])
    return result * 0.5 + 0.5


def worley3d(p: ArrayLike) -> FloatArray:
    p = np.asarray(p, float)
    ip, fp = np.floor(p), frac(p)
    distance_squared = np.full(p.shape[:-1], np.inf)
    for x in range(-1, 2):
        for y in range(-1, 2):
            for z in range(-1, 2):
                offset = np.array((x, y, z))
                difference = offset + hash33(ip + offset) - fp
                distance_squared = np.minimum(distance_squared, np.sum(difference**2, axis=-1))
    return saturate(1.0 - np.sqrt(distance_squared))


def simplex_noise3d(p: ArrayLike) -> FloatArray:
    """Ashima-style 3-D simplex noise, remapped to approximately [0, 1]."""
    p = np.asarray(p, float)
    # A compact simplex implementation using the shader's lattice/hash but
    # hash-derived gradients; its domain, attenuation, and output mapping match.
    i = np.floor(p + np.sum(p, axis=-1, keepdims=True) / 3.0)
    x0 = p - i + np.sum(i, axis=-1, keepdims=True) / 6.0
    rank = np.sum(x0[..., :, None] >= x0[..., None, :], axis=-1) - 1
    i1 = (rank >= 2).astype(float)
    i2 = (rank >= 1).astype(float)
    offsets = (np.zeros(3), i1, i2, np.ones(3))
    shifts = (0.0, 1.0 / 6.0, 1.0 / 3.0, 0.5)
    total = np.zeros(p.shape[:-1])
    for offset, shift in zip(offsets, shifts):
        x = x0 - offset + shift
        gradient = normalize(hash33(np.mod(i + offset, 289.0)) * 2.0 - 1.0)
        attenuation = np.maximum(0.5 - np.sum(x * x, axis=-1), 0.0)
        total += attenuation**4 * np.sum(gradient * x, axis=-1)
    return saturate(total * 52.5 + 0.5)


def open_simplex2s_approx3d(p: ArrayLike) -> FloatArray:
    rx, ry, rz = 2.03, 2.54, 1.72
    cx, sx, cy, sy, cz, sz = (
        np.cos(rx),
        np.sin(rx),
        np.cos(ry),
        np.sin(ry),
        np.cos(rz),
        np.sin(rz),
    )
    rxm = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
    rym = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
    rzm = np.array(((cz, -sz, 0), (sz, cz, 0), (0, 0, 1)))
    return simplex_noise3d(np.asarray(p, float) @ (rzm @ rym @ rxm).T)


def perlin_vec3(p: ArrayLike) -> FloatArray:
    p = np.asarray(p, float)
    return np.stack((perlin3d(p), perlin3d(p + 17.0), perlin3d(p + 37.0)), axis=-1) * 2.0 - 1.0


def curl(p: ArrayLike) -> FloatArray:
    p, e = np.asarray(p, float), 0.5773
    dx = perlin_vec3(p + (e, 0, 0)) - perlin_vec3(p - (e, 0, 0))
    dy = perlin_vec3(p + (0, e, 0)) - perlin_vec3(p - (0, e, 0))
    dz = perlin_vec3(p + (0, 0, e)) - perlin_vec3(p - (0, 0, e))
    return np.stack(
        (dy[..., 2] - dz[..., 1], dz[..., 0] - dx[..., 2], dx[..., 1] - dy[..., 0]),
        axis=-1,
    )


def curl_warped_noise(p: ArrayLike, warp_frequency: float) -> FloatArray:
    p = np.asarray(p, float)
    return simplex_noise3d(p + normalize(curl(p * warp_frequency) + 1e-4) * warp_frequency)


def base_noise(p: ArrayLike, algorithm: NoiseAlgorithm = "simplex", warp_frequency: float = 0.75) -> FloatArray:
    functions: dict[str, Callable[[ArrayLike], FloatArray]] = {
        "simplex": simplex_noise3d,
        "opensimplex2s": open_simplex2s_approx3d,
        "perlin": perlin3d,
        "value": value3d,
        "worley": worley3d,
    }
    if algorithm == "curl_domain_warp":
        return curl_warped_noise(p, warp_frequency)
    try:
        return functions[algorithm](p)
    except KeyError as error:
        raise ValueError(f"unknown noise algorithm: {algorithm!r}") from error


def fractal_noise(
    st: ArrayLike,
    noise_frequency: float = 1.0,
    noise_amplitude: float = 1.0,
    octaves: int = 4,
    warp_frequency: float = 0.75,
    algorithm: NoiseAlgorithm = "simplex",
) -> FloatArray:
    q = np.asarray(st, float) * noise_frequency
    result = np.zeros(q.shape[:-1])
    amplitude = 1.0
    normalization = 0.0
    for _ in range(max(0, int(octaves))):
        result += base_noise(q, algorithm, warp_frequency) * amplitude
        normalization += amplitude
        amplitude *= 0.5
        q *= 2.0
    if normalization:
        result /= normalization
    return saturate(result * noise_amplitude)
