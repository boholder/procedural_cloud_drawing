import numpy as np
import pytest

from src.cg_hlsl import dot, frac, length, lerp, normalize, saturate, smoothstep, step


@pytest.mark.parametrize(
    ("function", "value", "expected"),
    [
        (saturate, (-1.0, 0.25, 2.0), (0.0, 0.25, 1.0)),
        (frac, (-1.25, 0.25, 2.75), (0.75, 0.25, 0.75)),
    ],
)
def test_scalar_array_intrinsics(function, value, expected) -> None:
    np.testing.assert_allclose(function(value), expected)


def test_lerp_supports_scalar_and_per_element_t() -> None:
    np.testing.assert_allclose(lerp(10.0, 20.0, 0.25), 12.5)
    np.testing.assert_allclose(lerp((0.0, 10.0), (10.0, 20.0), (0.0, 0.5)), (0.0, 15.0))


def test_step_is_zero_below_edge_and_one_at_edge() -> None:
    np.testing.assert_array_equal(step(0.5, (0.25, 0.5, 0.75)), (0.0, 1.0, 1.0))


def test_smoothstep_has_expected_endpoints_and_midpoint() -> None:
    np.testing.assert_allclose(smoothstep(0.0, 1.0, (-1.0, 0.0, 0.5, 1.0, 2.0)), (0.0, 0.0, 0.5, 1.0, 1.0))


def test_smoothstep_handles_equal_edges_without_nan() -> None:
    result = smoothstep(1.0, 1.0, (0.0, 1.0, 2.0))

    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, (0.0, 0.0, 1.0))


def test_dot_and_length_support_batched_vectors() -> None:
    vectors = np.array(((3.0, 4.0), (5.0, 12.0)))

    np.testing.assert_allclose(dot(vectors, (3.0, 4.0)), (25.0, 63.0))
    np.testing.assert_allclose(dot((3.0, 4.0), (3.0, 4.0)), 25.0)
    np.testing.assert_allclose(length(vectors), (5.0, 13.0))


def test_normalize_returns_unit_vectors() -> None:
    result = normalize(((3.0, 4.0), (0.0, 2.0)))

    np.testing.assert_allclose(result, ((0.6, 0.8), (0.0, 1.0)))
    np.testing.assert_allclose(length(result), 1.0)


def test_normalize_zero_vector_is_finite_and_zero() -> None:
    result = normalize((0.0, 0.0, 0.0))

    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result, (0.0, 0.0, 0.0))
