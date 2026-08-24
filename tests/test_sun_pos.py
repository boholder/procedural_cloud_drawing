import math

import pytest

from src.sun_pos import sun_altitude_azimuth, sun_direction

# Conventional dates in a non-leap year.
MARCH_EQUINOX = 80  # March 21
JUNE_SOLSTICE = 172  # June 21
SEPTEMBER_EQUINOX = 266  # September 23
DECEMBER_SOLSTICE = 355  # December 21

# day_of_year, hours_of_day, location, expected_alt_az, expected_direction
TestCase = tuple[int, float, tuple[float, float], tuple[float, float | None], tuple[float, float, float]]

CASES: list[TestCase] = [
    (MARCH_EQUINOX, 12.0, (0.0, 0.0), (90.0, None), (0.0, 1.0, 0.0)),
    (
        JUNE_SOLSTICE,
        12.0,
        (0.0, 0.0),
        (66.56, 0.0),
        (
            0.0,
            math.sin(math.radians(66.56)),
            math.cos(math.radians(66.56)),
        ),
    ),
    (SEPTEMBER_EQUINOX, 12.0, (0.0, 0.0), (90.0, None), (0.0, 1.0, 0.0)),
    (
        DECEMBER_SOLSTICE,
        12.0,
        (0.0, 0.0),
        (66.56, 180.0),
        (
            0.0,
            math.sin(math.radians(66.56)),
            -math.cos(math.radians(66.56)),
        ),
    ),
]


@pytest.mark.parametrize(
    (
        "day_of_year",
        "time_of_day_hours",
        "location",
        "expected_alt_az",
    ),
    [pytest.param(_[0], _[1], _[2], _[3]) for _ in CASES],
)
def test_alt_az(
    day_of_year: int,
    location: tuple[float, float],
    time_of_day_hours: float,
    expected_alt_az: tuple[float, float | None],
) -> None:
    latitude, longitude = location
    expected_altitude, expected_azimuth = expected_alt_az
    altitude, azimuth = sun_altitude_azimuth(
        day_of_year=day_of_year,
        time_of_day_hours=time_of_day_hours,
        latitude_deg=latitude,
        longitude_deg=longitude,
    )

    assert altitude == pytest.approx(expected_altitude, abs=2.5)
    if expected_azimuth is not None:
        assert azimuth == pytest.approx(expected_azimuth, abs=2.5)


@pytest.mark.parametrize(
    ("day_of_year", "time_of_day_hours", "location", "expected_direction"),
    [pytest.param(_[0], _[1], _[2], _[4]) for _ in CASES],
)
def test_direction(
    day_of_year: int,
    location: tuple[float, float],
    time_of_day_hours: float,
    expected_direction: tuple[float, float, float],
) -> None:
    latitude, longitude = location
    east, up, north = sun_direction(
        day_of_year=day_of_year,
        time_of_day_hours=time_of_day_hours,
        latitude_deg=latitude,
        longitude_deg=longitude,
    )

    # check normalization
    assert math.sqrt(sum(component * component for component in (east, up, north))) == pytest.approx(1.0, abs=1e-12)

    expected_east, expected_up, expected_north = expected_direction
    assert east == pytest.approx(expected_east, abs=0.06)
    assert up == pytest.approx(expected_up, abs=0.06)
    assert north == pytest.approx(expected_north, abs=0.06)
