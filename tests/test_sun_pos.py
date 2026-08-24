import math

import pytest

from src.sun_pos import sun_altitude_azimuth, sun_direction

AltitudeAzimuth = tuple[float, float | None]
Direction = tuple[float, float, float]
Expectation = tuple[AltitudeAzimuth, Direction]
ExpectationsOfOneDay = tuple[int, Expectation, Expectation, Expectation, Expectation]


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
    return (720 - equation_of_time_minutes(day_of_year) - 4.0 * longitude + 60 * utc_offset_hours) / 60


def midnight(day_of_year: int, longitude: float) -> float:
    """Return local standard-clock time for solar midnight."""
    return (noon(day_of_year, longitude) + 12.0) % 24.0


def dawn(day_of_year: int, longitude: float) -> float:
    """Return local standard-clock time for apparent-solar 06:00."""
    return (noon(day_of_year, longitude) - 6.0) % 24.0


def dusk(day_of_year: int, longitude: float) -> float:
    """Return local standard-clock time for apparent-solar 18:00."""
    return (noon(day_of_year, longitude) + 6.0) % 24.0


def reverse_noon_night_dir(case: Expectation) -> Expectation:
    (altitude, azimuth), (east, up, north) = case
    reversed_azimuth = (azimuth + 180) % 360 if azimuth else None
    return (-altitude, reversed_azimuth), (-east, -up, -north)


def reverse_dawn_dusk_dir(case: Expectation) -> Expectation:
    """Mirror a dawn direction across north/south to obtain the dusk direction."""
    (altitude, azimuth), (east, up, north) = case
    dusk_azimuth = None if azimuth is None else (360 - azimuth) % 360
    return (altitude, dusk_azimuth), (-east, up, north)


def altitude_azimuth_case(altitude_deg: float, azimuth_deg: float) -> Expectation:
    altitude_rad = math.radians(altitude_deg)
    azimuth_rad = math.radians(azimuth_deg)
    horizontal_length = math.cos(altitude_rad)
    return (
        (altitude_deg, azimuth_deg),
        (
            horizontal_length * math.sin(azimuth_rad),
            math.sin(altitude_rad),
            horizontal_length * math.cos(azimuth_rad),
        ),
    )


# Conventional dates in a non-leap year.
MARCH_EQUINOX = 80  # March 21
JUNE_SOLSTICE = 172  # June 21
SEPTEMBER_EQUINOX = 266  # September 23
DECEMBER_SOLSTICE = 355  # December 21

# Azimuths
NORTH = 0
EAST = 90
SOUTH = 180
WEST = 270

# Altitudes
# 23.4334 degrees
TROPIC = 23 + 26 / 60
TROPIC_RAD = math.radians(TROPIC)
# 66.5667 degrees
SUB_TROPIC_RAD = math.radians(90 - TROPIC)
# 43.1334 degrees
SUB_2_TROPIC_RAD = math.radians(90 - 2 * TROPIC)
# At solar 06:00/18:00 on a solstice at a Tropic:
# altitude = asin(sin(TROPIC)^2) ~= 9.1 degrees,
TROPIC_SOLSTICE_ALT_DEG = math.degrees(math.asin(math.sin(TROPIC_RAD) ** 2))
# the dawn azimuth = atan(1 / sin(TROPIC)) ~= 68.3 degrees.
TROPIC_SOLSTICE_AZ_DEG = math.degrees(math.atan(1 / math.sin(TROPIC_RAD)))

# Locations
NULL_ISLAND = (0, 0)
TROPIC_NORTH = (TROPIC, 0)
TROPIC_SOUTH = (-TROPIC, 0)
EAST_P = (0, 90)
WEST_P = (0, -90)
NORTH_POLE = (90, 0)
SOUTH_POLE = (-90, 0)

# Altitude Azimuth and Direction pairs
# same alt-az must be the same direction
NOON_HIGHEST = ((90, None), (0, 1, 0))
NOON_HIGHEST_NORTH = ((90 - TROPIC, NORTH), (0, math.sin(SUB_TROPIC_RAD), math.cos(SUB_TROPIC_RAD)))
NOON_HIGHEST_SOUTH = ((90 - TROPIC, SOUTH), (0, math.sin(SUB_TROPIC_RAD), -math.cos(SUB_TROPIC_RAD)))
NOON_LOWEST_NORTH = (
    (90 - 2 * TROPIC, NORTH),
    (0, math.sin(SUB_2_TROPIC_RAD), math.cos(SUB_2_TROPIC_RAD)),
)
NOON_LOWEST_SOUTH = (
    (90 - 2 * TROPIC, SOUTH),
    (0, math.sin(SUB_2_TROPIC_RAD), -math.cos(SUB_2_TROPIC_RAD)),
)
MIDNIGHT_LOWEST = reverse_noon_night_dir(NOON_HIGHEST)
MIDNIGHT_SOUTH = reverse_noon_night_dir(NOON_HIGHEST_NORTH)
MIDNIGHT_NORTH = reverse_noon_night_dir(NOON_HIGHEST_SOUTH)
MIDNIGHT_LOWEST_SOUTH = reverse_noon_night_dir(NOON_LOWEST_NORTH)
MIDNIGHT_LOWEST_NORTH = reverse_noon_night_dir(NOON_LOWEST_SOUTH)

JUN_NOON_NORTH_POLE = ((TROPIC, SOUTH), (0, math.sin(TROPIC_RAD), -math.cos(TROPIC_RAD)))
JUN_MIDN_NORTH_POLE = ((TROPIC, NORTH), (0, math.sin(TROPIC_RAD), math.cos(TROPIC_RAD)))
JUN_DAWN_NORTH_POLE = ((TROPIC, EAST), (math.cos(TROPIC_RAD), math.sin(TROPIC_RAD), 0))
JUN_DUSK_NORTH_POLE = reverse_dawn_dusk_dir(JUN_DAWN_NORTH_POLE)
EQU_NOON_NORTH_POLE = ((0, SOUTH), (0, 0, -1))
EQU_MIDN_NORTH_POLE = ((0, NORTH), (0, 0, 1))
EQU_DAWN_NORTH_POLE = altitude_azimuth_case(0, EAST)
EQU_DUSK_NORTH_POLE = reverse_dawn_dusk_dir(EQU_DAWN_NORTH_POLE)
SEP_DAWN_NORTH_POLE = altitude_azimuth_case(0, EAST)
SEP_DUSK_NORTH_POLE = reverse_dawn_dusk_dir(SEP_DAWN_NORTH_POLE)
DEC_NOON_NORTH_POLE = ((-TROPIC, SOUTH), (0, -math.sin(TROPIC_RAD), -math.cos(TROPIC_RAD)))
DEC_MIDN_NORTH_POLE = ((-TROPIC, NORTH), (0, -math.sin(TROPIC_RAD), math.cos(TROPIC_RAD)))
DEC_DAWN_NORTH_POLE = ((-TROPIC, EAST), (math.cos(TROPIC_RAD), -math.sin(TROPIC_RAD), 0))
DEC_DUSK_NORTH_POLE = reverse_dawn_dusk_dir(DEC_DAWN_NORTH_POLE)

JUN_NOON_SOUTH_POLE = ((TROPIC, NORTH), (0, math.sin(TROPIC_RAD), math.cos(TROPIC_RAD)))
JUN_MIDN_SOUTH_POLE = ((TROPIC, SOUTH), (0, math.sin(TROPIC_RAD), -math.cos(TROPIC_RAD)))
JUN_DAWN_SOUTH_POLE = ((-TROPIC, EAST), (math.cos(TROPIC_RAD), -math.sin(TROPIC_RAD), 0))
JUN_DUSK_SOUTH_POLE = reverse_dawn_dusk_dir(JUN_DAWN_SOUTH_POLE)
EQU_NOON_SOUTH_POLE = ((0, NORTH), (0, 0, 1))
EQU_MIDN_SOUTH_POLE = ((0, SOUTH), (0, 0, -1))
EQU_DAWN_SOUTH_POLE = altitude_azimuth_case(0, EAST)
EQU_DUSK_SOUTH_POLE = reverse_dawn_dusk_dir(EQU_DAWN_SOUTH_POLE)
SEP_DAWN_SOUTH_POLE = altitude_azimuth_case(0, EAST)
SEP_DUSK_SOUTH_POLE = reverse_dawn_dusk_dir(SEP_DAWN_SOUTH_POLE)
DEC_NOON_SOUTH_POLE = ((-TROPIC, NORTH), (0, -math.sin(TROPIC_RAD), math.cos(TROPIC_RAD)))
DEC_MIDN_SOUTH_POLE = ((-TROPIC, SOUTH), (0, -math.sin(TROPIC_RAD), -math.cos(TROPIC_RAD)))
DEC_DAWN_SOUTH_POLE = ((TROPIC, EAST), (math.cos(TROPIC_RAD), math.sin(TROPIC_RAD), 0))
DEC_DUSK_SOUTH_POLE = reverse_dawn_dusk_dir(DEC_DAWN_SOUTH_POLE)

JUN_DAWN_EQUATOR = ((0, 90 - TROPIC), (math.cos(TROPIC_RAD), 0, math.sin(TROPIC_RAD)))
JUN_DUSK_EQUATOR = reverse_dawn_dusk_dir(JUN_DAWN_EQUATOR)
EQU_DAWN = ((0, EAST), (1, 0, 0))
EQU_DUSK = reverse_dawn_dusk_dir(EQU_DAWN)
SEP_DAWN_EQUATOR = altitude_azimuth_case(0, EAST)
DEC_DAWN_EQUATOR = ((0, 90 + TROPIC), (math.cos(TROPIC_RAD), 0, -math.sin(TROPIC_RAD)))
DEC_DUSK_EQUATOR = reverse_dawn_dusk_dir(DEC_DAWN_EQUATOR)

JUN_DAWN_TROPIC_NORTH = altitude_azimuth_case(TROPIC_SOLSTICE_ALT_DEG, TROPIC_SOLSTICE_AZ_DEG)
JUN_DUSK_TROPIC_NORTH = reverse_dawn_dusk_dir(JUN_DAWN_TROPIC_NORTH)
EQU_DAWN_TROPIC_NORTH = altitude_azimuth_case(0, 90)
EQU_DUSK_TROPIC_NORTH = reverse_dawn_dusk_dir(EQU_DAWN_TROPIC_NORTH)
DEC_DAWN_TROPIC_NORTH = altitude_azimuth_case(-TROPIC_SOLSTICE_ALT_DEG, SOUTH - TROPIC_SOLSTICE_AZ_DEG)
DEC_DUSK_TROPIC_NORTH = reverse_dawn_dusk_dir(DEC_DAWN_TROPIC_NORTH)

JUN_DAWN_TROPIC_SOUTH = altitude_azimuth_case(-TROPIC_SOLSTICE_ALT_DEG, TROPIC_SOLSTICE_AZ_DEG)
JUN_DUSK_TROPIC_SOUTH = reverse_dawn_dusk_dir(JUN_DAWN_TROPIC_SOUTH)
EQU_DAWN_TROPIC_SOUTH = altitude_azimuth_case(0, 90)
EQU_DUSK_TROPIC_SOUTH = reverse_dawn_dusk_dir(EQU_DAWN_TROPIC_SOUTH)
DEC_DAWN_TROPIC_SOUTH = altitude_azimuth_case(TROPIC_SOLSTICE_ALT_DEG, SOUTH - TROPIC_SOLSTICE_AZ_DEG)
DEC_DUSK_TROPIC_SOUTH = reverse_dawn_dusk_dir(DEC_DAWN_TROPIC_SOUTH)

# day_of_year, hours_of_day, location, expected_alt_az, expected_direction
TestCase = tuple[int, float, tuple[float, float], AltitudeAzimuth, Direction]


def gen_cases(
    location: tuple[float, float],
    cases: tuple[ExpectationsOfOneDay, ...],
) -> list[TestCase]:
    longitude = location[1]
    generated_cases: list[TestCase] = []
    for day, noon_expectation, midnight_expectation, dawn_expectation, dusk_expectation in cases:
        for time, (altitude_azimuth, direction) in (
            (noon(day, longitude), noon_expectation),
            (midnight(day, longitude), midnight_expectation),
            (dawn(day, longitude), dawn_expectation),
            (dusk(day, longitude), dusk_expectation),
        ):
            generated_cases.append((day, time, location, altitude_azimuth, direction))
    return generated_cases


# day_of_year, noon_expectation, midnight_expectation, dawn_expectation, dusk_expectation
EQUATOR_CASES: tuple[ExpectationsOfOneDay, ...] = (
    (MARCH_EQUINOX, NOON_HIGHEST, MIDNIGHT_LOWEST, EQU_DAWN, EQU_DUSK),
    (JUNE_SOLSTICE, NOON_HIGHEST_NORTH, MIDNIGHT_NORTH, JUN_DAWN_EQUATOR, JUN_DUSK_EQUATOR),
    (SEPTEMBER_EQUINOX, NOON_HIGHEST, MIDNIGHT_LOWEST, SEP_DAWN_EQUATOR, EQU_DUSK),
    (DECEMBER_SOLSTICE, NOON_HIGHEST_SOUTH, MIDNIGHT_SOUTH, DEC_DAWN_EQUATOR, DEC_DUSK_EQUATOR),
)
TROPIC_NORTH_CASES: tuple[ExpectationsOfOneDay, ...] = (
    (JUNE_SOLSTICE, NOON_HIGHEST, MIDNIGHT_LOWEST_NORTH, JUN_DAWN_TROPIC_NORTH, JUN_DUSK_TROPIC_NORTH),
    (SEPTEMBER_EQUINOX, NOON_HIGHEST_SOUTH, MIDNIGHT_NORTH, EQU_DAWN_TROPIC_NORTH, EQU_DUSK_TROPIC_NORTH),
    (DECEMBER_SOLSTICE, NOON_LOWEST_SOUTH, MIDNIGHT_LOWEST, DEC_DAWN_TROPIC_NORTH, DEC_DUSK_TROPIC_NORTH),
)
TROPIC_SOUTH_CASES: tuple[ExpectationsOfOneDay, ...] = (
    (JUNE_SOLSTICE, NOON_LOWEST_NORTH, MIDNIGHT_LOWEST, JUN_DAWN_TROPIC_SOUTH, JUN_DUSK_TROPIC_SOUTH),
    (SEPTEMBER_EQUINOX, NOON_HIGHEST_NORTH, MIDNIGHT_SOUTH, EQU_DAWN_TROPIC_SOUTH, EQU_DUSK_TROPIC_SOUTH),
    (DECEMBER_SOLSTICE, NOON_HIGHEST, MIDNIGHT_LOWEST_SOUTH, DEC_DAWN_TROPIC_SOUTH, DEC_DUSK_TROPIC_SOUTH),
)
NORTH_POLE_CASES: tuple[ExpectationsOfOneDay, ...] = (
    (MARCH_EQUINOX, EQU_NOON_NORTH_POLE, EQU_MIDN_NORTH_POLE, EQU_DAWN_NORTH_POLE, EQU_DUSK_NORTH_POLE),
    (JUNE_SOLSTICE, JUN_NOON_NORTH_POLE, JUN_MIDN_NORTH_POLE, JUN_DAWN_NORTH_POLE, JUN_DUSK_NORTH_POLE),
    (
        SEPTEMBER_EQUINOX,
        EQU_NOON_NORTH_POLE,
        EQU_MIDN_NORTH_POLE,
        SEP_DAWN_NORTH_POLE,
        SEP_DUSK_NORTH_POLE,
    ),
    (
        DECEMBER_SOLSTICE,
        DEC_NOON_NORTH_POLE,
        DEC_MIDN_NORTH_POLE,
        DEC_DAWN_NORTH_POLE,
        DEC_DUSK_NORTH_POLE,
    ),
)
SOUTH_POLE_CASES: tuple[ExpectationsOfOneDay, ...] = (
    (MARCH_EQUINOX, EQU_NOON_SOUTH_POLE, EQU_MIDN_SOUTH_POLE, EQU_DAWN_SOUTH_POLE, EQU_DUSK_SOUTH_POLE),
    (JUNE_SOLSTICE, DEC_NOON_SOUTH_POLE, DEC_MIDN_SOUTH_POLE, JUN_DAWN_SOUTH_POLE, JUN_DUSK_SOUTH_POLE),
    (
        SEPTEMBER_EQUINOX,
        EQU_NOON_SOUTH_POLE,
        EQU_MIDN_SOUTH_POLE,
        SEP_DAWN_SOUTH_POLE,
        SEP_DUSK_SOUTH_POLE,
    ),
    (
        DECEMBER_SOLSTICE,
        JUN_NOON_SOUTH_POLE,
        JUN_MIDN_SOUTH_POLE,
        DEC_DAWN_SOUTH_POLE,
        DEC_DUSK_SOUTH_POLE,
    ),
)

CASES: list[TestCase] = [
    *gen_cases(NULL_ISLAND, EQUATOR_CASES),
    *gen_cases(EAST_P, EQUATOR_CASES),
    *gen_cases(WEST_P, EQUATOR_CASES),
    *gen_cases(TROPIC_NORTH, TROPIC_NORTH_CASES),
    *gen_cases(TROPIC_SOUTH, TROPIC_SOUTH_CASES),
    *gen_cases(NORTH_POLE, NORTH_POLE_CASES),
    *gen_cases(SOUTH_POLE, SOUTH_POLE_CASES),
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

    assert altitude == pytest.approx(expected_altitude, abs=0.4)
    if expected_azimuth is not None:
        assert azimuth == pytest.approx(expected_azimuth, abs=0.4) or azimuth == pytest.approx(
            expected_azimuth + 360, abs=0.4
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
    assert math.sqrt(sum(component * component for component in (east, up, north))) == pytest.approx(1, abs=1e-12)

    expected_east, expected_up, expected_north = expected_direction
    assert east == pytest.approx(expected_east, abs=0.1)
    assert up == pytest.approx(expected_up, abs=0.1)
    assert north == pytest.approx(expected_north, abs=0.1)
