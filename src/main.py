from pathlib import Path

from src.core import SkySettings, render_sky, save_sky


def main() -> None:
    output = "sky_preview.png"
    image = render_sky(width=800, height=400, settings=SkySettings())
    save_sky(output, image)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
