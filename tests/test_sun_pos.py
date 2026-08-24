import math
from typing import Any

import pytest

from src.sun_pos import sun_altitude_azimuth, sun_direction

# Conventional dates in a non-leap year.
MARCH_EQUINOX = 80  # March 21
JUNE_SOLSTICE = 172  # June 21
SEPTEMBER_EQUINOX = 266  # September 23
DECEMBER_SOLSTICE = 355  # December 21


def equation_of_time_minutes(day_of_year) -> float:
    g = 2.0 * math.pi * (day_of_year - 1) / 365.25

    return 229.18 * (
        0.000075
        + 0.001868 * math.cos(g)
        - 0.032077 * math.sin(g)
        - 0.014615 * math.cos(2.0 * g)
        - 0.040849 * math.sin(2.0 * g)
    )


def noon(day_of_year: int, longitude: float) -> float:
    utc_offset_hours = round(longitude / 15.0)
    return (720 - equation_of_time_minutes(day_of_year) - 4.0 * longitude + 60.0 * utc_offset_hours) / 60.0


def midnight(day_of_year: int, longitude: float) -> float:
    """Return local standard-clock time for solar midnight."""
    return (noon(day_of_year, longitude) + 12.0) % 24.0


def reverse_dir(case: tuple[tuple[float, float | None], tuple[float, float, float]]) -> tuple:
    (altitude, azimuth), (east, up, north) = case
    reversed_azimuth = (azimuth + 180.0) % 360.0 if azimuth else None
    return (-altitude, reversed_azimuth), (-east, -up, -north)


def midnight(day_of_year: int, longitude_east_deg: float) -> float:
    """Return local standard-clock time for solar midnight."""
    return (noon(day_of_year, longitude_east_deg) + 12.0) % 24.0


def reverse_dir(case: tuple[tuple[float, float | None], tuple[float, float, float]]) -> tuple:
    (altitude, azimuth), (east, up, north) = case
    reversed_azimuth = (azimuth + 180.0) % 360.0 if azimuth else None
    return (-altitude, reversed_azimuth), (-east, -up, -north)


# Altitudes
# 23.4334 degrees
TROPIC = 23 + 26 / 60
# 66.5667 degrees
SUB_TROPIC_RAD = math.radians(90 - TROPIC)
# 43.1334 degrees
SUB_2_TROPIC_RAD = math.radians(90 - 2 * TROPIC)
TROPIC_RAD = math.radians(TROPIC)

# Azimuths
NORTH = 0.0
EAST = 90.0
SOUTH = 180.0
WEST = 270.0

# Locations
NULL_ISLAND = (0.0, 0.0)
TROPIC_NORTH = (TROPIC, 0.0)
TROPIC_SOUTH = (-TROPIC, 0.0)
EAST_P = (0.0, 90)
WEST_P = (0.0, -90)
NORTH_POLE = (90, 0)
SOUTH_POLE = (-90, 0)

# Altitude Azimuth and Direction pairs
# same alt-az must be the same direction
HIGH_NOON = ((90.0, None), (0.0, 1.0, 0.0))
NOON_HIGHEST_NORTH = ((90 - TROPIC, NORTH), (0.0, math.sin(SUB_TROPIC_RAD), math.cos(SUB_TROPIC_RAD)))
NOON_HIGHEST_SOUTH = ((90 - TROPIC, SOUTH), (0.0, math.sin(SUB_TROPIC_RAD), -math.cos(SUB_TROPIC_RAD)))
NOON_LOWEST_NORTH = (
    (90 - 2 * TROPIC, NORTH),
    (0.0, math.sin(SUB_2_TROPIC_RAD), math.cos(SUB_2_TROPIC_RAD)),
)
NOON_LOWEST_SOUTH = (
    (90 - 2 * TROPIC, SOUTH),
    (0.0, math.sin(SUB_2_TROPIC_RAD), -math.cos(SUB_2_TROPIC_RAD)),
)
NORTH_POLE_SUMMER = ((TROPIC, SOUTH), (0.0, math.sin(TROPIC_RAD), -math.cos(TROPIC_RAD)))
SOUTH_POLE_SUMMER = ((TROPIC, NORTH), (0.0, math.sin(TROPIC_RAD), math.cos(TROPIC_RAD)))
MIDNIGHT = reverse_dir(HIGH_NOON)
MIDNIGHT_SOUTH = reverse_dir(NOON_HIGHEST_NORTH)
MIDNIGHT_NORTH = reverse_dir(NOON_HIGHEST_SOUTH)
MIDNIGHT_LOW_SOUTH = reverse_dir(NOON_LOWEST_NORTH)
MIDNIGHT_LOW_NORTH = reverse_dir(NOON_LOWEST_SOUTH)

# day_of_year, hours_of_day, location, expected_alt_az, expected_direction
TestCase = tuple[int, float, tuple[float, float], Any]

CASES: list[TestCase] = [
    (MARCH_EQUINOX, noon(MARCH_EQUINOX, 0.0), NULL_ISLAND, *HIGH_NOON),
    (JUNE_SOLSTICE, noon(JUNE_SOLSTICE, 0.0), NULL_ISLAND, *NOON_HIGHEST_NORTH),
    (SEPTEMBER_EQUINOX, noon(SEPTEMBER_EQUINOX, 0.0), NULL_ISLAND, *HIGH_NOON),
    (DECEMBER_SOLSTICE, noon(DECEMBER_SOLSTICE, 0.0), NULL_ISLAND, *NOON_HIGHEST_SOUTH),
    (JUNE_SOLSTICE, noon(JUNE_SOLSTICE, 0.0), TROPIC_NORTH, *HIGH_NOON),
    (SEPTEMBER_EQUINOX, noon(SEPTEMBER_EQUINOX, 0.0), TROPIC_NORTH, *NOON_HIGHEST_SOUTH),
    (DECEMBER_SOLSTICE, noon(DECEMBER_SOLSTICE, 0.0), TROPIC_NORTH, *NOON_LOWEST_SOUTH),
    (DECEMBER_SOLSTICE, noon(DECEMBER_SOLSTICE, 0.0), TROPIC_SOUTH, *HIGH_NOON),
    (SEPTEMBER_EQUINOX, noon(SEPTEMBER_EQUINOX, 0.0), TROPIC_SOUTH, *NOON_HIGHEST_NORTH),
    (JUNE_SOLSTICE, noon(JUNE_SOLSTICE, 0.0), TROPIC_SOUTH, *NOON_LOWEST_NORTH),
    (MARCH_EQUINOX, noon(MARCH_EQUINOX, EAST_P[1]), EAST_P, *HIGH_NOON),
    (MARCH_EQUINOX, noon(MARCH_EQUINOX, WEST_P[1]), WEST_P, *HIGH_NOON),
    (JUNE_SOLSTICE, noon(JUNE_SOLSTICE, NORTH_POLE[1]), NORTH_POLE, *NORTH_POLE_SUMMER),
    (DECEMBER_SOLSTICE, noon(DECEMBER_SOLSTICE, SOUTH_POLE[1]), SOUTH_POLE, *SOUTH_POLE_SUMMER),
    (MARCH_EQUINOX, midnight(MARCH_EQUINOX, 0.0), NULL_ISLAND, *MIDNIGHT),
    (JUNE_SOLSTICE, midnight(JUNE_SOLSTICE, 0.0), NULL_ISLAND, *MIDNIGHT_NORTH),
    (SEPTEMBER_EQUINOX, midnight(SEPTEMBER_EQUINOX, 0.0), NULL_ISLAND, *MIDNIGHT),
    (DECEMBER_SOLSTICE, midnight(DECEMBER_SOLSTICE, 0.0), NULL_ISLAND, *MIDNIGHT_SOUTH),
    (JUNE_SOLSTICE, midnight(JUNE_SOLSTICE, 0.0), TROPIC_NORTH, *MIDNIGHT_LOW_NORTH),
    (SEPTEMBER_EQUINOX, midnight(SEPTEMBER_EQUINOX, 0.0), TROPIC_NORTH, *MIDNIGHT_NORTH),
    (DECEMBER_SOLSTICE, midnight(DECEMBER_SOLSTICE, 0.0), TROPIC_NORTH, *MIDNIGHT),
    (DECEMBER_SOLSTICE, midnight(DECEMBER_SOLSTICE, 0.0), TROPIC_SOUTH, *MIDNIGHT_LOW_SOUTH),
    (SEPTEMBER_EQUINOX, midnight(SEPTEMBER_EQUINOX, 0.0), TROPIC_SOUTH, *MIDNIGHT_SOUTH),
    (JUNE_SOLSTICE, midnight(JUNE_SOLSTICE, 0.0), TROPIC_SOUTH, *MIDNIGHT),
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

    assert altitude == pytest.approx(expected_altitude, abs=0.3)
    if expected_azimuth is not None:
        assert azimuth == pytest.approx(expected_azimuth, abs=0.3) or azimuth == pytest.approx(
            expected_azimuth + 360, abs=0.3
        )


# noinspection PyTypeChecker
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
    assert east == pytest.approx(expected_east, abs=0.1)
    assert up == pytest.approx(expected_up, abs=0.1)
    assert north == pytest.approx(expected_north, abs=0.1)
