# imgui_manim

Short [manim](https://www.manim.community) animations that explain the steps of a neuroscience imaging pipeline, from motion correction to the voltage in a single dendrite. Each one is a script built on a numpy model or a stored run, so the traces and numbers on screen are computed, not drawn by hand.

![Signal quality (d')](docs/animations/dprime.gif)

## Run one

You need [uv](https://docs.astral.sh/uv/), ffmpeg, and the cairo and pango libraries manim draws with. No LaTeX is needed.

```bash
sudo pacman -S uv ffmpeg cairo pango                                         # Arch
sudo apt install ffmpeg build-essential python3-dev libcairo2-dev libpango1.0-dev   # Debian, Ubuntu; uv from its site
brew install uv ffmpeg cairo pango pkg-config                                # macOS
```

The text is set in Noto Sans. Without that font Pango substitutes another, and labels shift.

```bash
git clone https://github.com/FlynnOConnell/imgui_manim
cd imgui_manim
uv run -p 3.12 animations/dprime.py --out out/dprime
```

Each script declares its own dependencies, so `uv run` installs them on the first run. The run prints d' for each simulated cell, then the files it wrote to `--out`: one PNG per chapter, `dprime.mp4` and `dprime.gif`. It takes about 45 s on a desktop CPU.

| option | effect |
|--------|--------|
| `--out <dir>` | the output folder, required |
| `--stills-only` | stops after the PNGs, in about 4 s |
| `--width <px>` | width of the mp4 and the PNGs, 1280 by default |
| `--gif-width <px>` | width of the gif, 960 by default |

Every script in `animations/` runs this way, except where its section below says otherwise. manim's dependency pydub prints `SyntaxWarning: invalid escape sequence` on Python 3.12; the render is unaffected.

## Animations

The name under each animation is its script in `animations/`.

### Imaging pipeline

Each of these has a page in [imgui_ophys](https://github.com/FlynnOConnell/imgui_ophys), where you work the same step by hand on the same model.

<table>
<tr>
<td width="50%"><img src="docs/animations/motion.gif" alt="Motion correction"><br><code>motion</code>: registering every frame to a reference image</td>
<td width="50%"><img src="docs/animations/nmf.gif" alt="Demixing"><br><code>nmf</code>: a movie as footprints x traces, fitted by HALS</td>
</tr>
<tr>
<td><img src="docs/animations/traces.gif" alt="Traces"><br><code>traces</code>: from a spike to a recorded dF/F</td>
<td><img src="docs/animations/dprime.gif" alt="Signal quality"><br><code>dprime</code>: a measure of how well spikes stand out from noise</td>
</tr>
<tr>
<td><img src="docs/animations/svd.gif" alt="SVD"><br><code>svd</code>: an image as a sum of rank-1 pieces</td>
<td><img src="docs/animations/silo.gif" alt="SILo"><br><code>silo</code>: synapses found from their events, then their traces fitted</td>
</tr>
</table>

### One real cell

These four show one pyramidal cell from the MICrONS dataset. The cell and the synapse are real and every voltage is simulated.

<table>
<tr>
<td width="50%"><img src="docs/animations/synapse_to_soma.gif" alt="A synapse and a soma burst"><br><code>synapse_to_soma</code>: four glutamate releases and a soma burst, as the traces each leaves at spine, branch and soma</td>
<td width="50%"><img src="docs/animations/epsp.gif" alt="EPSP"><br><code>epsp</code>: one release, and inputs summing until the branch fires a dendritic spike</td>
</tr>
<tr>
<td><img src="docs/animations/bap.gif" alt="bAP"><br><code>bap</code>: a spike running back up the dendrites, and how it opens NMDA receptors</td>
<td><img src="docs/animations/subcellular.gif" alt="Subthreshold voltage"><br><code>subcellular</code>: a dendrite's potential below the level that fires a spike</td>
</tr>
</table>

Cell images: MICrONS Consortium 2025, [www.microns-explorer.org](https://www.microns-explorer.org), CC BY 4.0, not this repository's license. Their [source, terms and limits](src/imgui_manim/lessons/synapse_to_soma/neuron/README.md) sit beside the images.

### Methods in detail

<table>
<tr>
<td width="50%"><img src="docs/animations/strip_registration.gif" alt="StripRegistration"><br><code>strip_registration</code>: a thin strip drifts over a dendrite and each aligned frame joins the template</td>
<td width="50%"><img src="docs/animations/silo_activity.gif" alt="SILo activity image"><br><code>silo_activity</code>: SILo's activity image building up, where only the synapses accumulate</td>
</tr>
<tr>
<td><img src="docs/animations/compression.gif" alt="masknmf compression"><br><code>compression</code>: masknmf's compression and denoising, from a real masknmf run</td>
<td></td>
</tr>
</table>

`masknmf_pipeline` has no gif. It is masknmf's pipeline as five plain manim scenes, from a movie to its traces:

```bash
uv run -p 3.12 animations/masknmf_pipeline.py [<scene> ...] --out <dir> [-q high_quality]
```

### Miller Brain Studio

The Studio is the viewer in [pml_utilities](https://github.com/FlynnOConnell/pml_utilities), run as `mbo`. Every format and feature on screen is the package's own.

<table>
<tr>
<td width="50%"><img src="docs/animations/graphical_abstract.gif" alt="Graphical abstract"><br><code>graphical_abstract</code>: the files the Studio reads and writes, and the pipelines it runs</td>
<td width="50%"><img src="docs/animations/suite2p_demo.gif" alt="suite2p demo"><br><code>suite2p_demo</code>: the Studio's own windows, from a raw movie to suite2p's cells and traces</td>
</tr>
<tr>
<td><img src="docs/animations/open_scrub_save.gif" alt="Open, scrub, save"><br><code>open_scrub_save</code>: a movie opened, binned and saved as an OME-Zarr</td>
<td></td>
</tr>
</table>

The movie is `demoMovie.tif` from [CaImAn](https://github.com/flatironinstitute/CaImAn), which this repository's license does not cover. `graphical_abstract` and `open_scrub_save` read it from `~/data`:

```bash
mkdir -p ~/data
curl -L -o ~/data/demoMovie.tif https://raw.githubusercontent.com/flatironinstitute/CaImAn/main/example_movies/demoMovie.tif
```

## Write your own

`plugins/manim-animation/skills/manim-animation/` holds a copy of the kit that needs only manim, numpy and pillow, and `example.py`, a three-chapter animation in under 80 lines: a trace, a number, and noise added while the number counts down. Copy both files into your project and edit the example.

```bash
cd plugins/manim-animation/skills/manim-animation
uv run -p 3.12 example.py --out out/example
```

### With Claude

The same folder is a [Claude Code](https://claude.com/claude-code) skill. `SKILL.md` holds the rules these animations were built to: the chapter arc, the text and picture rules, the manim traps, and a review checklist. Install it from inside Claude Code:

```
/plugin marketplace add FlynnOConnell/imgui_manim
/plugin install manim-animation@imgui-manim
```

Or copy the folder, with no plugin:

```bash
cp -r plugins/manim-animation/skills/manim-animation ~/.claude/skills/
```

Then ask for an animation, naming the method and the model: "make a manim animation of how a running-percentile baseline gives dF/F, from `model.py`". Claude writes the script on the kit, renders the stills, and shows them to you. Send back a list of cuts and it re-renders.

## Building an animation

An animation is a `build()` that returns nested AnimationGroups, on `animations/kit.py`: manim as a drawing engine with a Cairo camera and no `Scene`. `kit.timeline` begins every leaf in time order and `seek(t)` scrubs to any second, so one build gives the stills, the mp4 and the gif. `Count` makes a number react to a morph, and `Play` and `Recolor` step an image or a grid through frames.

Each chapter adds one idea. Titles name the content ("Add noise", "Truncated SVD") or are absent, traces carry scale bars instead of axes, and labels state the thing and its provenance ("Ground truth spikes (simulated)").

Models live in two places. An animation with an imgui_ophys page reads that page's model, so a model change reaches both. `kit.lesson_model(name)` finds it in the installed `imgui_ophys`, and looks in `src/imgui_manim/lessons/<name>/model.py` first for animations that have no page. The colors come from imgui_ophys too. imgui_ophys knows nothing about this repository.

To try a change in a local imgui_ophys checkout:

```bash
uv run -p 3.12 --with-editable ../imgui_ophys animations/dprime.py --out out/dprime
```

## Stored data and the window

`animations/data/` and `src/imgui_manim/lessons/` hold the runs and images some animations draw. [docs/data.md](docs/data.md) says where each came from and how to regenerate it.

`imgui-manim-window` draws these animations live in an imgui window, on your own movie: a timeline scrubs each page while the movie plays inside it. See [docs/window.md](docs/window.md).

![The Overview page on demoMovie.tif](docs/animations/masknmf_overview.gif)

## Tests

```bash
uv venv -p 3.12 && uv pip install -e ".[dev]"
uv run --no-sync pytest
```

With the `window` extra installed, `pytest` also runs the window's tests, and one of them opens a hidden window.

See [ARCHITECTURE.md](ARCHITECTURE.md) for how the scrubbing works.

## License

The code is [MIT](LICENSE). The MICrONS images are CC BY 4.0 and the demo movie is CaImAn's, as noted above.
