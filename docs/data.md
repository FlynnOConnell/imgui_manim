# Stored data

Most animations simulate what they draw. The ones here read data stored in this repository. Each section says where that data came from and how to regenerate it.

## compression

`src/imgui_manim/lessons/compression/run.npz` holds a real masknmf run on a simulated two-photon movie. To regenerate it, use an environment with [masknmf](https://github.com/FlynnOConnell/masknmf-toolbox), torch, matplotlib and [pml_utilities](https://github.com/FlynnOConnell/pml_utilities). It takes about 7 min on a GPU. Training is not deterministic, so the numbers shift a little. Delete `~/.cache/imgui_manim/compression_simulated` to compress again.

```bash
python animations/compression_run.py
```

The same animation runs on a recording. The exporter reads any file `mbo_utilities.imread` opens, registers it, compresses it into a cache, and exports one block. A recording has no true traces, so the animation compares against the raw trace and the closing table reports noise and gain relative to raw.

```bash
python animations/compression_run.py --input <movie> --block auto --cache <dir> --out <npz> [--uint16] [--name "<name on screen>"]
uv run -p 3.12 animations/compression.py --run <npz> --out <dir> --width 1920
```

`--block auto` chooses the block whose kept candidates look most like one cell: a skewed trace over a compact, centered image, in a block where the second pass keeps more than the first and every kept candidate fits the four rows drawn. It writes `<npz>_blocks.png` with the best blocks; pass one with `--block y,x`. Compression is cached, so another block takes about 90 s. `--uint16` is for Bruker exports, which store uint16 pixels as int16.

## suite2p_demo and open_scrub_save

`animations/data/suite2p_demo/` holds the Studio's real windows and a suite2p run on `demoMovie.tif`. `animations/data/open_scrub_save/save_as.json` holds the numbers `open_scrub_save` shows. The scripts in `scripts/studio/` regenerate them, in an environment with pml_utilities and suite2p:

```bash
python scripts/studio/run_suite2p.py
python scripts/studio/export_suite2p.py
python scripts/studio/capture_windows.py
python scripts/studio/export_save_as.py
```

## The MICrONS cell

`src/imgui_manim/lessons/synapse_to_soma/neuron/` holds the panels of the cell, the electron micrograph of its synapse, and the path distance of every pixel from the soma and from Spine 1. The volumes are public and need no account. `fetch_microns_cell.py` downloads the 240 MB mesh to `~/data/microns_pyramidal`, and `export_neuron.py` writes the folder, copying the panels from imgui_ophys.

```bash
uv run -p 3.12 scripts/synapse_to_soma/fetch_microns_cell.py
uv run -p 3.12 scripts/synapse_to_soma/export_neuron.py
```

## masknmf_pipeline

`animations/data/masknmf_pipeline/run.npz` is one masknmf run on a simulated movie. It cannot be regenerated, because its exporter no longer exists.
