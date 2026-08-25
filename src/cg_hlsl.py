"""Small NumPy equivalents of CG/HLSL intrinsics used by the shader."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.floating]
_EPSILON = 1e-12


def saturate(x: ArrayLike) -> FloatArray:
    return np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)


def frac(x: ArrayLike) -> FloatArray:
    return np.mod(np.asarray(x, dtype=np.float32), 1.0)


def lerp(a: ArrayLike, b: ArrayLike, t: ArrayLike) -> FloatArray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    return a + (b - a) * t


def step(edge: ArrayLike, x: ArrayLike) -> FloatArray:
    return (np.asarray(x) >= np.asarray(edge)).astype(np.float32)


def smoothstep(edge0: ArrayLike, edge1: ArrayLike, x: ArrayLike) -> FloatArray:
    edge0 = np.asarray(edge0, dtype=np.float32)
    edge1 = np.asarray(edge1, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)

    denominator = edge1 - edge0
    safe_denominator = np.where(
        np.abs(denominator) < _EPSILON,
        np.where(denominator < 0.0, -_EPSILON, _EPSILON),
        denominator,
    )

    t = saturate((x - edge0) / safe_denominator)
    return t * t * (3.0 - 2.0 * t)


def dot(a: ArrayLike, b: ArrayLike) -> FloatArray:
    return np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32))


def length(v: ArrayLike) -> FloatArray:
    return np.linalg.norm(np.asarray(v, dtype=np.float32), axis=-1)


def normalize(v: ArrayLike, epsilon: float = _EPSILON) -> FloatArray:
    value = np.asarray(v, dtype=np.float32)
    magnitude = np.linalg.norm(value, axis=-1, keepdims=True)
    return value / np.maximum(magnitude, epsilon)
