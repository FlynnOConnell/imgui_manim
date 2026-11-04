"""
Screenshots for the README: the window's own framebuffer, from a hidden window (`--screenshot`), one process per
page. Saves docs/window/<page>.png, or under --out. Needs a desktop session for the GL context.

    python scripts/window/screenshot.py --movie demoMovie.tif --fs 10              # every page
    python scripts/window/screenshot.py --movie demoMovie.tif --fs 10 --page raw   # one
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SHOTS = {"overview": 9.5, "raw": 9.5, "background": 13.0, "residual": 10.0}  # page: seconds into its story, the end


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--movie", required=True, help="a .tif or .npy movie (frames, height, width)")
    parser.add_argument("--fs", required=True, help="the movie's frame rate in Hz")
    parser.add_argument("--page", choices=list(SHOTS), default=None, help="default: every page")
    parser.add_argument("--out", type=Path, default=REPO / "docs" / "window")
    args = parser.parse_args()
    for page in [args.page] if args.page else list(SHOTS):
        filepath_png = args.out / f"{page}.png"
        command = [sys.executable, "-m", "imgui_manim.window.app", "--movie", args.movie, "--fs", args.fs, "--scale", "1",
                   "--page", page, "--time", str(SHOTS[page]), "--screenshot", str(filepath_png)]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(filepath_png)


if __name__ == "__main__":
    main()
