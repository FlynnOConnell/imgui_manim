# The window

`imgui-manim-window` explains how masknmf's compression splits a movie into three movies:

    registered = background + signal + residual

Each page is one manim animation, built from the movie itself and drawn by manim's Cairo camera into an imgui-bundle window. A timeline scrubs the animation forward or back while the movie plays inside its panels. Click a pixel to follow it on every page.

![The Overview animation on demoMovie.tif](animations/masknmf_overview.gif)

## Run it

```bash
uv venv -p 3.12 && uv pip install -e ".[dev,window]"
uv run --no-sync imgui-manim-window --movie <tif or npy> --fs <Hz> [--negative]
```

`--negative` flips a movie from an indicator that dims with activity. masknmf comes at the version pml_utilities pins. manim draws with cairo and pango, so pycairo and manimpango build from source against the system libraries on Linux.

Under the animation are the timeline (drag to scrub), the chapters (click one to play from its start), the current chapter's paragraph with this movie's numbers in it, and the movie's frame slider. Space plays and pauses, left and right step chapters, up and down step pages, Esc goes to Overview, and Ctrl +, Ctrl - and Ctrl 0 resize.

## The three movies

masknmf's `CompressionArray` rebuilds a registered frame as

    noise SD x (U V) + mean image + spatial trend x temporal trend

and `decomposition.split_pmd` names the parts:

| piece | contents | in masknmf |
|-------|----------|------------|
| background | the mean image plus a slow trend (the maximin spline detrender) | `mean_image + spatial_trend_basis @ temporal_trend_basis` |
| signal | noise SD x U V: a few smooth components per block, blended | `noise_variance_image * (spatial_compressed @ temporal_compressed)` |
| residual | registered - background - signal | `CompressionResidualArray` |

## Pages

| page | chapters |
|------|----------|
| [Overview](window/overview.png) | the registered movie shrinks into the first of four panels: = background + signal + residual; four traces per followed pixel: the picked one, then the highest signal SNR |
| [Raw movie](window/raw.png) | a stack of recorded frames; recorded beside registered; one pixel's trace, recorded and registered |
| [Background](window/background.png) | the mean image; the trend at one pixel; the background movie; registered - background, the fluctuations |
| [Residual](window/residual.png) | registered - background - signal; the residual at one pixel against 2 noise SDs; its histogram over every pixel against a normal curve |

The window has no page for compression's blocks and roughness test. The `compression` animation covers those.

## Images

```bash
python scripts/window/screenshot.py --movie <tif> --fs <Hz>                  # docs/window/<page>.png
python scripts/window/overview_gif.py --movie <tif> --fs <Hz>                # docs/animations/masknmf_overview.gif
python scripts/window/demo.py --movie <tif> --fs <Hz> --out tour.mp4         # the real window on a scripted tour
```
