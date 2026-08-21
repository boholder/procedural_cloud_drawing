import argparse
from pathlib import Path

from src.core import SkySettings, render_sky, save_sky


def inner(
    camera_direction: tuple[float, float, float] = (0.0, 0.0, 1.0),
    width: int = 800,
    height: int = 400,
    output: str | Path = "preview.png",
) -> int:
    output = Path(output)
    image = render_sky(
        camera_direction=camera_direction,
        width=width,
        height=height,
        settings=SkySettings(),
    )
    save_sky(output, image)
    print(f"wrote {output}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a procedural sky camera view")
    parser.add_argument(
        "-d",
        "--camera-direction",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 1.0),
        help="world-space direction at the center of the image",
    )
    parser.add_argument("-w", "--width", type=int, default=800)
    parser.add_argument("-hh", "--height", type=int, default=400)
    parser.add_argument("-o", "--output", type=Path, default=Path("sky_preview.png"))
    args = parser.parse_args()

    cd: tuple[float, float, float] = (
        args.camera_direction[0],
        args.camera_direction[1],
        args.camera_direction[2],
    )
    return inner(
        camera_direction=cd,
        width=args.width,
        height=args.height,
        output=args.output,
    )


def hd_main() -> int:
    return inner(width=1280, height=720, output="preview.png")


def four_k_main() -> int:
    return inner(width=3840, height=2160, output="preview.png")


if __name__ == "__main__":
    main()
