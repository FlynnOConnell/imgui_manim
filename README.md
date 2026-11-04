# imgui_manim

manim animations of imaging pipeline steps. This repo describes a method; its companion `imgui_ophys` lets you work the
same method by hand, one page per step. Both read one numpy model, so a model change reaches the animation and the page
together.

Each animation is a script in `animations/` built on `animations/kit.py` (manim as a drawing engine with a Cairo camera,
no `Scene`). `kit.lesson_model(name)` loads `src/imgui_manim/lessons/<name>/model.py`, or imgui_ophys's model of that
name when this repo has none, so `imgui_ophys` must be checked out beside this repo. It stands in empty modules for the
lesson packages, so their `__init__` files (which import imgui_bundle) never run.

An animation is a `build()` returning nested AnimationGroups. `kit.timeline` begins every leaf in time order and
`seek(t)` scrubs to any second, so the same build gives the stills, the mp4 and the gif. `Count` makes a number react to
a morph; `Play` and `Recolor` step an image or a grid through frames.

## imgui_ophys pages

One animation per page, drawn from the page's model. Each chapter adds one idea. Titles name the content ("Add noise",
"Truncated SVD") or are absent, traces carry scale bars instead of axes, labels state the thing and its provenance
("Ground truth spikes (simulated)"), and a number counts along as the picture morphs.

![Motion correction](docs/animations/motion.gif)

![Demixing](docs/animations/nmf.gif)

![Traces](docs/animations/traces.gif)

![Signal quality](docs/animations/dprime.gif)

![SVD](docs/animations/svd.gif)

![SILo](docs/animations/silo.gif)

![Subcellular voltage](docs/animations/subcellular.gif)

```
uv run -p 3.12 animations/<page>.py --out <dir> [--stills-only] [--gif-width 720]
```

The scripts declare their own dependencies (manim, numpy, pillow). Each writes an mp4, a gif and a PNG per chapter. The
gifs here are 720 px at 10 fps. `subcellular` needs imgui_ophys's `worktree-subcellular` branch until it merges, because
its model is only there.

## strip_registration

A thin strip drifts over a dendrite and StripRegistration adds each aligned frame to the template where it landed. A padding pixel turns blue as frames cover it and joins the template at 100.

![strip_registration](docs/animations/strip_registration.gif)

```
uv run -p 3.12 animations/strip_registration.py --out <dir> [--stills-only] [--gif-width 720]
```

Needs `uv` and `ffmpeg` on PATH. Model tests: `pytest`.

## silo_activity

SILo's activity image builds up on a simulated strip: each space-time maximum of the matched-filtered movie adds its square to its pixel, and only the synapses accumulate. It uses the SILo page's model; `silo` above is the whole method.

![silo_activity](docs/animations/silo_activity.gif)

```
uv run -p 3.12 animations/silo_activity.py --out <dir> [--stills-only] [--gif-width 720]
```

## compression

masknmf's compression and its denoising pass on a simulated two-photon movie: blocks, block SVD, the roughness test against noise, the blind-spot network, the second pass, and a closing table of correlation and amplitude against the true traces. The figures come from a real masknmf run stored in `src/imgui_manim/lessons/compression/run.npz`.

![compression](docs/animations/compression.gif)

```
uv run -p 3.12 animations/compression.py --out <dir> [--stills-only] [--gif-width 720]
```

To regenerate `run.npz` (about 7 min on a GPU; training is not deterministic, so the numbers shift a little; delete `~/.cache/imgui_manim/compression_simulated` to compress again):

```
PYTHONPATH=~/repos/masknmf-toolbox ~/envs/pml/.venv/bin/python animations/compression_run.py
```

The same animation runs on a recording. The exporter reads any file `mbo_utilities.imread` opens (`pml_utilities` is a dependency for this), registers it, compresses it into a cache, and exports one block. A recording has no true traces, so the animation compares against the raw trace and the closing table reports noise and gain relative to raw.

```
PYTHONPATH=~/repos/masknmf-toolbox ~/envs/pml/.venv/bin/python animations/compression_run.py \
    --input <movie> --block auto --cache <dir> --out <npz> [--uint16] [--name "<name on screen>"]
uv run -p 3.12 animations/compression.py --run <npz> --out <dir> --width 1920
```

`--block auto` chooses the block whose kept candidates look most like one cell: a skewed trace over a compact, centered image, in a block where the second pass keeps more than the first and every kept candidate fits the four rows drawn. It writes `<npz>_blocks.png` with the best blocks; pass one with `--block y,x`. Compression is cached, so another block takes about 90 s. `--uint16` is for Bruker exports, which store uint16 pixels as int16.
