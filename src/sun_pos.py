"""Approximate solar direction for a Unity world/skybox.

Coordinate convention of returned vector:
    x = East
    y = Up
    z = North

The returned vector points FROM the world origin / observer TO the visible Sun.
For the Celestia Flow shader in this project, use the same vector as the
skybox sun direction. For a Unity Directional Light, orient its forward
(direction light travels) to -sun_direction.

Inputs:
    day_of_year: 1..365 (or 366 in a leap year)
    time_of_day_hours: local civil clock time, e.g. 13.5 = 13:30
    latitude_deg: north positive, south negative
    longitude_deg: east positive, west negative
    The standard UTC offset is approximated from longitude.

This is a NOAA-style approximation. It omits terrain/obstacle horizon,
refraction, daylight-saving-time rules, and sub-minute astronomical effects.
"""

from __future__ import annotations

import math

float3 = tuple[float, float, float]


def sun_direction(
    day_of_year: int,
    time_of_day_hours: float,
    latitude_deg: float,
    longitude_deg: float,
    days_in_year: int = 365,
) -> float3:
    """Return a normalized (x,y,z)=(east, up, north) vector from world center pointing toward the sun.

    `time_of_day_hours` is local *standard* clock time. If using a time zone
    with daylight saving time, pass standard time, or subtract one hour from
    the displayed daylight-saving clock value before calling this function.

    ref: General Solar Position Calculations https://gml.noaa.gov/grad/solcalc/solareqns.PDF
    """
    altitude, azimuth = _sun_altitude_azimuth_radians(
        day_of_year,
        time_of_day_hours,
        latitude_deg,
        longitude_deg,
        days_in_year,
    )

    # Alt/Az -> Unity world direction: y+=Up, z+=North, x+=East, z-=South, x-=West.
    horizontal_length = math.cos(altitude)
    north = horizontal_length * math.cos(azimuth)
    up = math.sin(altitude)
    east = horizontal_length * math.sin(azimuth)

    return east, up, north


def _sun_altitude_azimuth_radians(
    day_of_year: int,
    time_of_day_hours: float,
    latitude_deg: float,
    longitude_deg: float,
    days_in_year: int = 365,
) -> tuple[float, float]:
    """Calculate solar altitude and azimuth in radians."""
    if not 1 <= day_of_year <= days_in_year:
        raise ValueError(f"day_of_year must be in 1..{days_in_year}")
    if not 0.0 <= time_of_day_hours < 24.0:
        raise ValueError("time_of_day_hours must be in [0, 24)")
    if not -90.0 <= latitude_deg <= 90.0:
        raise ValueError("latitude_deg must be in [-90, 90]")
    if not -180.0 <= longitude_deg <= 180.0:
        raise ValueError("longitude_deg must be in [-180, 180]")

    gamma = (2.0 * math.pi / days_in_year) * (day_of_year - 1 + (time_of_day_hours - 12.0) / 24.0)

    equation_of_time_minutes = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2.0 * gamma)
        - 0.040849 * math.sin(2.0 * gamma)
    )

    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2.0 * gamma)
        + 0.000907 * math.sin(2.0 * gamma)
        - 0.002697 * math.cos(3.0 * gamma)
        + 0.001480 * math.sin(3.0 * gamma)
    )

    # Local standard clock time -> true solar time.
    clock_minutes = time_of_day_hours * 60.0
    utc_offset_hours = round(longitude_deg / 15.0)
    time_offset_minutes = equation_of_time_minutes + 4.0 * longitude_deg - 60.0 * utc_offset_hours

    true_solar_time_minutes = (clock_minutes + time_offset_minutes) % 1440.0

    # Negative before solar noon; positive after solar noon.
    hour_angle = math.radians(true_solar_time_minutes / 4.0 - 180.0)
    latitude = math.radians(latitude_deg)

    # Altitude above the horizon and azimuth clockwise from north:
    # North=0°, East=90°, South=180°, West=270°.
    sin_altitude = math.sin(latitude) * math.sin(declination) + math.cos(latitude) * math.cos(declination) * math.cos(
        hour_angle
    )
    altitude = math.asin(max(-1.0, min(1.0, sin_altitude)))

    azimuth = (
        math.atan2(
            math.sin(hour_angle),
            math.cos(hour_angle) * math.sin(latitude) - math.tan(declination) * math.cos(latitude),
        )
        + math.pi
    )
    azimuth %= 2.0 * math.pi
    return altitude, azimuth


def sun_altitude_azimuth(
    day_of_year: int,
    time_of_day_hours: float,
    latitude_deg: float,
    longitude_deg: float,
    days_in_year: int = 365,
) -> tuple[float, float]:
    """Return (altitude_deg, azimuth_deg), with azimuth clockwise from north."""
    altitude, azimuth = _sun_altitude_azimuth_radians(
        day_of_year,
        time_of_day_hours,
        latitude_deg,
        longitude_deg,
        days_in_year,
    )
    return math.degrees(altitude), math.degrees(azimuth)
