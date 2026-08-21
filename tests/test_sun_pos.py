import math

import pytest

from src.sun_pos import sun_altitude_azimuth, sun_direction

# Conventional dates in a non-leap year.
MARCH_EQUINOX = 80  # March 21
JUNE_SOLSTICE = 172  # June 21
SEPTEMBER_EQUINOX = 266  # September 23
DECEMBER_SOLSTICE = 355  # December 21


@pytest.mark.parametrize(
    (
        "day_of_year",
        "latitude",
        "longitude",
        "time_of_day_hours",
        "expected_altitude",
        "expected_azimuth",
    ),
    [
        pytest.param(MARCH_EQUINOX, 0.0, 0.0, 12.0, 90.0, None, id="march-equinox"),
        pytest.param(JUNE_SOLSTICE, 0.0, 0.0, 12.0, 66.56, 0.0, id="june-solstice"),
        pytest.param(SEPTEMBER_EQUINOX, 0.0, 0.0, 12.0, 90.0, None, id="september-equinox"),
        pytest.param(DECEMBER_SOLSTICE, 0.0, 0.0, 12.0, 66.56, 180.0, id="december-solstice"),
    ],
)
def test_seasonal_noon_position(
    day_of_year: int,
    latitude: float,
    longitude: float,
    time_of_day_hours: float,
    expected_altitude: float,
    expected_azimuth: float | None,
) -> None:
    """Check the four seasonal landmarks at 0° latitude, 0° longitude.

    These use latitude=0 and longitude=0. At the equator, the Sun is nearly
    straight overhead at local 12:00 on an equinox, north at the June solstice,
    and south at the December solstice.
    """
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
    ("day_of_year", "latitude", "longitude", "time_of_day_hours", "expected_direction"),
    [
        # (x, y, z) = (north, up, east)
        pytest.param(MARCH_EQUINOX, 0.0, 0.0, 12.0, (0.0, 1.0, 0.0), id="march-equinox"),
        pytest.param(
            JUNE_SOLSTICE, 0.0, 0.0, 12.0, (1.0, math.sin(math.radians(66.56)), 0.0), id="june-solstice"
        ),
        pytest.param(SEPTEMBER_EQUINOX, 0.0, 0.0, 12.0, (0.0, 1.0, 0.0), id="september-equinox"),
        pytest.param(
            DECEMBER_SOLSTICE,
            0.0,
            0.0,
            12.0,
            (-1.0, math.sin(math.radians(66.56)), 0.0),
            id="december-solstice",
        ),
    ],
)
def test_equator_greenwich_vector_has_expected_hemisphere(
    day_of_year: int,
    latitude: float,
    longitude: float,
    time_of_day_hours: float,
    expected_direction: tuple[float, float, float],
) -> None:
    """The vector is normalized and lies in the expected north/south half."""
    north, up, east = sun_direction(
        day_of_year=day_of_year,
        time_of_day_hours=time_of_day_hours,
        latitude_deg=latitude,
        longitude_deg=longitude,
    )

    assert math.sqrt(
        sum(component * component for component in (north, up, east))
    ) == pytest.approx(1.0, abs=1e-12)
    assert east == pytest.approx(0.0, abs=0.06)
    assert up > 0.90

    if day_of_year in (MARCH_EQUINOX, SEPTEMBER_EQUINOX):
        # Near zenith: horizontal direction is poorly defined, so test only
        # that both horizontal components are small.
        assert abs(north) < 0.06
    elif day_of_year == JUNE_SOLSTICE:
        assert north > 0.35
    else:
        assert north < -0.35
