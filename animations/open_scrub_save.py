# /// script
# requires-python = ">=3.11"
# dependencies = ["manim>=0.22", "numpy", "pillow<13", "tifffile", "matplotlib"]
# ///
"""
The Studio from a tif to an OME-Zarr, as one animation: `mbo ~/data/demoMovie.tif` opens the movie, the time slider
scrubs it, the Metadata Viewer shows the frame rate missing and Set Metadata supplies it, a window of 10 frames
becomes temporal binning with Apply to dataset, the frame rate follows, and File > Save as writes a sharded,
compressed OME-Zarr. Every widget is drawn as the Studio draws it; every number comes from mbo_utilities
(animations/data/open_scrub_save/save_as.json).

    uv run -p 3.12 animations/open_scrub_save.py --out <dir> [--stills-only] [--gif-width 720]

Text only, no LaTeX. Needs ffmpeg on PATH and ~/data/demoMovie.tif.
"""

import json
from pathlib import Path

import manim as mn
import numpy as np
import tifffile
from matplotlib import colormaps

import kit
from kit import ACCENT, DIM, EDGE, KEY, TEXT, Count, Play, at, caption, chapter, color, label, put, replace, timeline

FILEPATH_TIF = Path.home() / "data" / "demoMovie.tif"
FILEPATH_NUMBERS = Path(__file__).resolve().parent / "data" / "open_scrub_save" / "save_as.json"
N = json.loads(FILEPATH_NUMBERS.read_text())
FS, BIN = N["fs_hz"], N["frame_average"]

TITLE = "demoMovie.tif, binned by 10, to an OME-Zarr"
CHAPTERS = (
    ("", 6.0),
    ("Scrub frames", 6.0),
    ("Metadata Viewer (m)", 5.0),
    ("Set Metadata (Shift+M)", 5.5),
    ("Window Size 10", 7.0),
    ("Apply to dataset", 6.0),
    ("Metadata Viewer (m)", 4.0),
    ("File > Save as (s)", 5.0),
    ("Zarr Options", 7.0),
    ("", 10.0),
)
UNTITLED = {0: "mbo-demomovie-tif", 9: "on-disk"}

# the Studio's colors
WINDOW = (0.13, 0.14, 0.16)
STRIP = (0.17, 0.18, 0.21)
FIELD = (0.20, 0.30, 0.48)
HEADER = (0.80, 0.80, 0.20)
TAB = (0.26, 0.45, 0.75)
MKEY = (0.45, 0.75, 0.95)
REQUIRED = (1.00, 0.30, 0.30)
OK = (0.60, 0.80, 0.60)
PYRAMID = (0.60, 0.80, 1.00)
MONO = "Noto Sans Mono"

X_CANVAS, Y_CANVAS, H_CANVAS = -3.3, 0.45, 3.4
X_PANEL = 0.9  # left edge of the side panel's text
Y_SLIDER = -2.45
X_SLIDER0, X_SLIDER1 = -4.6, 4.3
FRAMES_PLAY = 120


# ----------------------------------------------------------------------
# the movie
# ----------------------------------------------------------------------


def load_movie() -> np.ndarray:
    return tifffile.imread(str(FILEPATH_TIF)).astype(np.float32)  ## (t, h, w)


def to_rgb(frames: np.ndarray, lo: float, hi: float) -> list:
    """Each (h, w) frame through gnuplot2, the Studio's canvas colormap, as (h, w, 3) uint8."""
    lut = (colormaps["gnuplot2"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    index = np.clip((frames - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return [lut[f] for f in index]


def running_mean(movie: np.ndarray, t: np.ndarray, size: int) -> np.ndarray:
    """The mean of the size frames ending at each t (fewer at the start), the Studio's mean projection."""
    return np.stack([movie[max(0, i - size + 1):i + 1].mean(axis=0) for i in t])


# ----------------------------------------------------------------------
# widgets, drawn as the Studio draws them
# ----------------------------------------------------------------------


def box(width: float, height: float, x: float, y: float, fill: tuple = STRIP, stroke: float = 0.0, c: tuple = EDGE) -> mn.RoundedRectangle:
    return put(mn.RoundedRectangle(corner_radius=0.05, width=width, height=height, fill_color=color(fill), fill_opacity=1.0,
                                   stroke_color=color(c), stroke_width=stroke), x, y)


def field(text: str, width: float, x: float, y: float, c: tuple = TEXT) -> mn.VGroup:
    """A value box with its text, left edge at x."""
    rect = box(width, 0.34, x + width / 2, y, FIELD)
    return mn.VGroup(rect, label(text, c, 16).move_to(rect))


def text_at(text: str, x: float, y: float, c: tuple = TEXT, size: float = 16) -> mn.Text:
    return label(text, c, size).move_to(at(x, y), aligned_edge=mn.LEFT)


def checkbox(x: float, y: float, checked: bool) -> mn.VGroup:
    """A box with its tick; unchecked, the tick is there at opacity 0 so a Transform can reveal it."""
    square = box(0.26, 0.26, x + 0.13, y, ACCENT if checked else FIELD)
    tick = mn.VMobject(stroke_color=color(TEXT), stroke_width=2.5).set_points_as_corners(
        [at(x + 0.06, y), at(x + 0.11, y - 0.06), at(x + 0.21, y + 0.07)])
    tick.set_stroke(opacity=1.0 if checked else 0.0)
    return mn.VGroup(square, tick)


def row(x: float, y: float, value: str, name: str, width: float = 0.75) -> mn.VGroup:
    """`[value] - + name`, the Studio's input row."""
    minus = box(0.26, 0.26, x + width + 0.22, y, FIELD)
    plus = box(0.26, 0.26, x + width + 0.54, y, FIELD)
    return mn.VGroup(field(value, width, x, y), minus, label("-", TEXT, 16).move_to(minus), plus, label("+", TEXT, 16).move_to(plus),
                     text_at(name, x + width + 0.8, y))


def menu_button(text: str, x: float, y: float, width: float, c: tuple = TEXT, fill: tuple = STRIP) -> mn.VGroup:
    rect = box(width, 0.34, x + width / 2, y, fill)
    return mn.VGroup(rect, label(text, c, 15).move_to(rect))


def metadata_panel(fs_text: str, fs_color: tuple, n_timepoints: int) -> mn.VGroup:
    """The Metadata Viewer over the side panel: its Imaging rows for this file."""
    x0, y0, w, h = 0.8, -2.0, 5.6, 4.4
    panel = put(mn.Rectangle(width=w, height=h, fill_color=color(WINDOW), fill_opacity=1.0, stroke_color=color(EDGE), stroke_width=1.5), x0 + w / 2, y0 + h / 2)
    rows = (("fs", fs_text, fs_color), ("Lx", "80 (px)", TEXT), ("Ly", "60 (px)", TEXT), ("num_zplanes", "1", TEXT), ("num_color_channels", "1", TEXT),
            ("num_timepoints", f"{n_timepoints}", TEXT), ("dtype", "'uint16'", TEXT))
    items = [panel, label("Metadata (?)", TEXT, 16).move_to(at(x0 + w / 2, y0 + h - 0.25)), text_at("Imaging", x0 + 0.2, y0 + h - 0.65, MKEY, 16)]
    for k, (key, value, c) in enumerate(rows):
        y = y0 + h - 1.0 - 0.33 * k
        items += [text_at(key, x0 + 0.2, y, MKEY, 15), text_at(value, x0 + 2.6, y, c, 15)]
    return mn.VGroup(*items)


def set_metadata_popup() -> mn.VGroup:
    x, y, w, h = -1.5, 0.2, 6.4, 2.3
    panel = put(mn.Rectangle(width=w, height=h, fill_color=color(WINDOW), fill_opacity=1.0, stroke_color=color(EDGE), stroke_width=1.5), x, y)
    title = label("Set Metadata", TEXT, 17).move_to(at(x, y + h / 2 - 0.25))
    name = text_at("Frame Rate", x - w / 2 + 0.3, y + 0.25, MKEY, 16)
    value = field(f"{FS:g}", 1.2, x - w / 2 + 2.2, y + 0.25)
    unit = text_at("Hz", x - w / 2 + 3.55, y + 0.25, DIM, 16)
    what = text_at("Volume (or frame) sampling rate in Hz.", x - w / 2 + 0.3, y - 0.25, DIM, 14)
    apply = menu_button("Apply", x - w / 2 + 0.3, y - h / 2 + 0.35, 1.2, TEXT, TAB)
    return mn.VGroup(panel, title, name, value, unit, what, apply)


def file_menu() -> mn.VGroup:
    x0, y_top = -6.3, 2.3
    items = (("Open File", "o"), ("Open Folder", "Shift+O"), ("Open Notebook", ""), ("Save as", "s"), ("Options", ""))
    w, hrow = 2.9, 0.36
    panel = put(mn.Rectangle(width=w, height=hrow * len(items) + 0.1, fill_color=color(STRIP), fill_opacity=1.0, stroke_color=color(EDGE), stroke_width=1.0),
                x0 + w / 2, y_top - (hrow * len(items) + 0.1) / 2)
    out = [panel]
    for k, (name, key) in enumerate(items):
        y = y_top - 0.25 - hrow * k
        if name == "Save as":
            out.append(box(w - 0.1, hrow - 0.04, x0 + w / 2, y, TAB))
        out += [text_at(name, x0 + 0.15, y, TEXT, 15), label(key, DIM, 14).move_to(at(x0 + w - 0.15, y), aligned_edge=mn.RIGHT)]
    return mn.VGroup(*out)


def save_dialog() -> tuple:
    """The Save As dialog with its zarr options; returns (dialog, unchecked boxes, checked boxes, save button)."""
    x, y, w, h = -1.5, -0.1, 9.6, 6.0
    L = x - w / 2 + 0.3
    panel = put(mn.Rectangle(width=w, height=h, fill_color=color(WINDOW), fill_opacity=1.0, stroke_color=color(EDGE), stroke_width=1.5), x, y)
    top = y + h / 2
    store = N["store_name"]
    size_mb = N["bytes_uncompressed"] / 1e6
    shape = N["binned"]["shape"]
    items = [panel, label("Save As", TEXT, 17).move_to(at(x, top - 0.25)),
             text_at("Output", L, top - 0.7, HEADER), field(".zarr", 1.3, L, top - 1.1), text_at("Ext", L + 1.45, top - 1.1, DIM),
             text_at("Selection", L, top - 1.6, HEADER),
             text_at("Timepoints", L, top - 2.0, TEXT), field(f"1:{shape[0]}", 1.6, L + 1.6, top - 2.0), text_at(f"{shape[0]}/{shape[0]}", L + 3.4, top - 2.0, DIM),
             text_at("Filename", L, top - 2.45, DIM, 15), text_at(store, L + 1.6, top - 2.45, OK, 15),
             text_at("Size", L, top - 2.75, DIM, 15), text_at(f"~{size_mb:.1f} MB (uncompressed)", L + 1.6, top - 2.75, TEXT, 15),
             text_at("Shape", L, top - 3.05, DIM, 15), text_at(f"({shape[0]}, {shape[-2]}, {shape[-1]})", L + 1.6, top - 3.05, TEXT, 15),
             text_at("Options", L + 5.1, top - 0.7, HEADER), *row(L + 5.1, top - 1.1, f"{BIN}", "Frame Average", 0.6).submobjects,
             text_at("Zarr Options", L + 5.1, top - 1.6, HEADER)]
    y_zarr = top - 2.0
    boxes_off, boxes_on = [], []
    for k, name in enumerate(("Sharded", "OME-Zarr")):
        boxes_off.append(checkbox(L + 5.1, y_zarr - 0.4 * k, False))
        boxes_on.append(checkbox(L + 5.1, y_zarr - 0.4 * k, True))
        items.append(text_at(name, L + 5.5, y_zarr - 0.4 * k, TEXT, 15))
    items += [text_at("Compression Level", L + 5.1, y_zarr - 0.85, TEXT, 15), field(f"{N['compression_level']}", 2.2, L + 5.1, y_zarr - 1.2),
              text_at("Pyramid (Multi-resolution)", L + 5.1, y_zarr - 1.7, PYRAMID, 15)]
    boxes_off.append(checkbox(L + 5.1, y_zarr - 2.1, False))
    boxes_on.append(checkbox(L + 5.1, y_zarr - 2.1, True))
    items += [text_at("Generate Pyramid", L + 5.5, y_zarr - 2.1, TEXT, 15),
              text_at("Max Levels", L + 5.1, y_zarr - 2.5, TEXT, 15), field(f"{N['pyramid_max_layers']}", 2.2, L + 5.1, y_zarr - 2.85)]
    save = menu_button("Save", L, y - h / 2 + 0.4, 1.4, TEXT, TAB)
    save_lit = menu_button("Save", L, y - h / 2 + 0.4, 1.4, TEXT, ACCENT)
    return mn.VGroup(*items), boxes_off, boxes_on, save, save_lit


def size_bar(mb: float, y: float, c: tuple) -> mn.VGroup:
    rect = put(mn.Rectangle(width=2.6 * mb, height=0.4, fill_color=color(c), fill_opacity=0.9, stroke_width=0), 0.5 + 1.3 * mb, y)
    return mn.VGroup(rect)


def size_label(mb: float) -> mn.Text:
    return label(f"{mb:.2f} MB", KEY, 30)


def readout_label(d: float) -> mn.Text:
    return label(f"d: {d:.0f}", TEXT, 15)


def count_label(v: float) -> mn.Text:
    return label(f"{v:.0f}", TEXT, 16)


def swap(titles: list, captions: list, k: int) -> list:
    """Chapter k's title and caption replace chapter k - 1's."""
    return [*replace(titles[k - 1], titles[k]), *replace(captions[k - 1], captions[k])]


# ----------------------------------------------------------------------
# the animation
# ----------------------------------------------------------------------


def build(seconds: list) -> kit.Timeline:
    movie = load_movie()
    n_t = movie.shape[0]
    lo, hi = np.percentile(movie[::20], 0.5), np.percentile(movie[::20], 99.8)
    t_scrub = np.linspace(0, n_t - 1, FRAMES_PLAY).astype(int)
    frames_raw = to_rgb(movie[t_scrub], lo, hi)
    mean_end = movie[n_t - BIN:].mean(axis=0)
    frames_cross = to_rgb(np.stack([(1 - a) * movie[-1] + a * mean_end for a in np.linspace(0, 1, 12)]), lo, hi)
    t_back = np.linspace(n_t - 1, BIN - 1, FRAMES_PLAY).astype(int)
    frames_mean_back = to_rgb(running_mean(movie, t_back, BIN), lo, hi)
    frames_binned = to_rgb(movie[:BIN].mean(axis=0)[None], lo, hi)
    level = N["levels"][0]
    mb_raw, mb_disk = N["bytes_uncompressed"] / 1e6, level["bytes_on_disk"] / 1e6

    title = put(label(TITLE, TEXT, 30), 0.0, 3.6)
    titles = kit.titles_for(CHAPTERS)

    # chapter 1: the command line, then the window
    terminal = put(mn.Rectangle(width=7.0, height=0.6, fill_color=color((0.05, 0.05, 0.06)), fill_opacity=1.0, stroke_color=color(EDGE), stroke_width=1.0), 0.0, 0.3)
    command = mn.Text("$ mbo ~/data/demoMovie.tif", font=MONO, font_size=22, color=color(TEXT)).move_to(terminal).align_to(terminal, mn.LEFT).shift(0.25 * mn.RIGHT)
    for glyph in command.get_family():
        if hasattr(glyph, "path_obj"):
            glyph.path_obj = None
    window = put(mn.Rectangle(width=13.4, height=5.9, fill_color=color(WINDOW), fill_opacity=1.0, stroke_color=color(EDGE), stroke_width=1.5), 0.0, -0.15)
    strip_top = box(13.2, 0.42, 0.0, 2.55, STRIP)
    menu = mn.VGroup(text_at("File", -6.4, 2.55, TEXT, 15), text_at("Docs", -5.6, 2.55, TEXT, 15),
                     menu_button("Metadata Viewer (m)", 3.5, 2.55, 2.9, TEXT, WINDOW))
    panel = kit.image(frames_raw[0], H_CANVAS, X_CANVAS, Y_CANVAS)
    canvas_name = label("demoMovie.tif", DIM, 15).next_to(panel, mn.UP, buff=0.12)
    tabs = mn.VGroup(menu_button("Preview", X_PANEL, 2.0, 1.2, TEXT, TAB), menu_button("Signal Quality", X_PANEL + 1.3, 2.0, 1.8, DIM, STRIP),
                     menu_button("Run", X_PANEL + 3.2, 2.0, 0.8, DIM, STRIP))
    side = mn.VGroup(text_at("Window Functions", X_PANEL, 1.5, HEADER), field("mean", 1.1, X_PANEL, 1.05), text_at("Projection", X_PANEL + 1.3, 1.05, TEXT),
                     *row(X_PANEL, 0.6, "1", "Window Size", 0.75).submobjects[1:], checkbox(X_PANEL, 0.12, False), text_at("Apply to dataset", X_PANEL + 0.4, 0.12, TEXT))
    window_size = label("1", TEXT, 16).move_to(at(X_PANEL + 0.12, 0.6), aligned_edge=mn.LEFT)
    window_size_box = box(0.75, 0.34, X_PANEL + 0.375, 0.6, FIELD)
    strip_bottom = box(13.2, 0.42, 0.0, -2.0, STRIP)
    controls_name = text_at("ImageWidget Controls", -6.4, -2.0, TEXT, 15)
    dim_name = text_at("dim 't:'", -6.4, Y_SLIDER, TEXT, 15)
    track = box(X_SLIDER1 - X_SLIDER0, 0.26, (X_SLIDER0 + X_SLIDER1) / 2, Y_SLIDER, FIELD)
    handle_w = 0.5
    handle = box(handle_w, 0.26, X_SLIDER0 + handle_w / 2, Y_SLIDER, (0.55, 0.70, 0.95))
    handle_end = box(handle_w, 0.26, X_SLIDER1 - handle_w / 2, Y_SLIDER, (0.55, 0.70, 0.95))
    handle_9 = box(handle_w, 0.26, X_SLIDER0 + handle_w / 2 + (X_SLIDER1 - X_SLIDER0 - handle_w) * (BIN - 1) / (n_t - 1), Y_SLIDER, (0.55, 0.70, 0.95))
    readout = label("d: 0", TEXT, 15).move_to(at(X_SLIDER1 + 0.3, Y_SLIDER), aligned_edge=mn.LEFT)
    readout_bin = label("d: 0", TEXT, 15).move_to(at(X_SLIDER1 + 0.3, Y_SLIDER), aligned_edge=mn.LEFT)
    chrome = [window, strip_top, menu, canvas_name, tabs, side, window_size_box, window_size, strip_bottom, controls_name, dim_name, track, handle, readout]

    # chapters 3, 4, 7: the metadata viewer twice, and the editor between
    meta_required = metadata_panel("Required", REQUIRED, n_t)
    meta_set = metadata_panel(f"{FS:.1f} (Hz)", TEXT, n_t)
    meta_binned = metadata_panel(f"{FS / BIN:.1f} (Hz)", TEXT, n_t // BIN)
    popup = set_metadata_popup()

    # chapters 5 and 6: the window and the binning readout
    apply_off, apply_on = side[-2], checkbox(X_PANEL, 0.12, True)
    averaged = text_at(f"{BIN} frames averaged", X_PANEL + 2.6, 0.12, OK, 15)
    frames_text = text_at(f"Frames: {n_t} -> {n_t // BIN}", X_PANEL, -0.3, TEXT, 15)
    rate_text = text_at(f"Rate: {FS:g} -> {FS / BIN:g} Hz", X_PANEL, -0.65, TEXT, 15)
    window_size_reset = label("1", TEXT, 16).move_to(at(X_PANEL + 0.12, 0.6), aligned_edge=mn.LEFT)

    # chapters 8 and 9: the menu and the dialog
    dropdown = file_menu()
    dialog, boxes_off, boxes_on, save, save_lit = save_dialog()

    # chapter 10: what is on disk
    tree_lines = (
        (f"{N['store_parent']}/", TEXT, 0.0),
        (f"{N['store_name']}/", TEXT, 0.4),
        ("zarr.json", MKEY, 0.8),
        (f"OME-NGFF {N['ome_version']}, axes {' '.join(N['axes'])}", DIM, 2.6),
        ("0/", MKEY, 0.8),
        (f"({', '.join(str(s) for s in level['shape'])}) {level['dtype']}", DIM, 2.6),
        ("c/0/0/0/0", MKEY, 1.2),
        (f"one shard of {level['shard_shape'][0]} frames, chunks of {level['chunk_shape'][0]} frame, {level['codecs'][-1]} {N['compression_level']}", DIM, 2.6),
    )
    tree = mn.VGroup()
    for k, (text, c, indent) in enumerate(tree_lines):
        tree.add(mn.Text(text, font=MONO, font_size=17, color=color(c)).move_to(at(-6.6 + 0.3 * indent, 2.4 - 0.45 * k), aligned_edge=mn.LEFT))
    for glyph in tree.get_family():
        if hasattr(glyph, "path_obj"):
            glyph.path_obj = None
    bar_name = text_at("uncompressed", 0.5, 0.9, DIM, 16)
    bar_disk_name = text_at("on disk", 0.5, 0.9, DIM, 16)
    bar = size_bar(mb_raw, 0.4, DIM)
    bar_disk = size_bar(mb_disk, 0.4, ACCENT)
    number = size_label(mb_raw).move_to(at(0.5, -0.3), aligned_edge=mn.LEFT)
    pyramid_text = mn.VGroup(text_at(f"Pyramid: {len(N['levels'])} level", 0.5, -1.4, TEXT, 18),
                             text_at(f"levels stop under {N['pyramid_min_size_px']} px; the frame is 60 x 80 px", 0.5, -1.8, DIM, 15))

    captions = [None, None,
                caption("fs is Required: the tif carries no frame interval."), None,
                caption("Projection mean over a window of 10 frames: display only."),
                caption("Every 10 frames become one; the window size resets to 1."),
                None, None, None, None]

    chapters = [
        chapter(seconds[0],
                mn.AnimationGroup(mn.FadeIn(title), mn.FadeIn(terminal), mn.FadeIn(command), lag_ratio=0.3, run_time=1.2),
                mn.AnimationGroup(mn.FadeOut(terminal), mn.FadeOut(command), mn.FadeIn(window, scale=0.95), run_time=0.8),
                mn.AnimationGroup(*[mn.FadeIn(m) for m in chrome[1:]], mn.FadeIn(panel), lag_ratio=0.03, run_time=1.5),
                lag_ratio=1.0),
        chapter(seconds[1], mn.AnimationGroup(mn.FadeOut(title), *swap(titles, captions, 1), run_time=0.8),
                mn.AnimationGroup(mn.Transform(handle, handle_end), Play(panel, frames_raw), Count(readout, 0, n_t - 1, readout_label, 0),
                                  run_time=4.0, rate_func=mn.linear),
                lag_ratio=1.0),
        chapter(seconds[2], mn.AnimationGroup(*swap(titles, captions, 2), run_time=0.8), mn.FadeIn(meta_required, shift=0.2 * mn.LEFT, run_time=0.8), lag_ratio=1.0),
        chapter(seconds[3], mn.AnimationGroup(*swap(titles, captions, 3), run_time=0.8), mn.FadeIn(popup, scale=0.95, run_time=0.6), mn.Wait(run_time=1.8),
                mn.AnimationGroup(mn.FadeOut(popup), *replace(meta_required, meta_set), run_time=0.8), lag_ratio=1.0),
        chapter(seconds[4], mn.AnimationGroup(*swap(titles, captions, 4), mn.FadeOut(meta_set), run_time=0.8),
                mn.AnimationGroup(Count(window_size, 1, BIN, count_label, 0), Play(panel, frames_cross), run_time=1.5),
                mn.AnimationGroup(mn.Transform(handle, handle_9), Play(panel, frames_mean_back), Count(readout, n_t - 1, BIN - 1, readout_label, 0),
                                  run_time=3.0, rate_func=mn.linear),
                lag_ratio=1.0),
        chapter(seconds[5], mn.AnimationGroup(*swap(titles, captions, 5), run_time=0.8),
                mn.AnimationGroup(mn.Transform(apply_off, apply_on), mn.FadeIn(averaged), run_time=0.5),
                mn.AnimationGroup(mn.FadeIn(frames_text), mn.FadeIn(rate_text), *replace(window_size, window_size_reset),
                                  Play(panel, frames_binned), *replace(readout, readout_bin), lag_ratio=0.3, run_time=1.5),
                lag_ratio=1.0),
        chapter(seconds[6], mn.AnimationGroup(*swap(titles, captions, 6), run_time=0.8), mn.FadeIn(meta_binned, shift=0.2 * mn.LEFT, run_time=0.8), lag_ratio=1.0),
        chapter(seconds[7], mn.AnimationGroup(*swap(titles, captions, 7), mn.FadeOut(meta_binned), run_time=0.8), mn.FadeIn(dropdown, run_time=0.5), mn.Wait(run_time=1.5),
                mn.AnimationGroup(mn.FadeOut(dropdown), mn.FadeIn(dialog, scale=0.97), *[mn.FadeIn(b) for b in boxes_off], mn.FadeIn(save), run_time=0.8),
                lag_ratio=1.0),
        chapter(seconds[8], mn.AnimationGroup(*swap(titles, captions, 8), run_time=0.8),
                mn.LaggedStart(*[mn.Transform(off, on) for off, on in zip(boxes_off, boxes_on)], lag_ratio=0.5, run_time=2.0),
                mn.AnimationGroup(*replace(save, save_lit), run_time=0.6),
                lag_ratio=1.0),
        chapter(seconds[9],
                mn.AnimationGroup(*swap(titles, captions, 9), *[mn.FadeOut(m) for m in (*chrome, panel, dialog, *boxes_off, save_lit, averaged, frames_text, rate_text,
                                                                    window_size_reset, readout_bin)], run_time=1.0),
                mn.AnimationGroup(mn.LaggedStart(*[mn.FadeIn(line) for line in tree], lag_ratio=0.2), mn.FadeIn(bar_name), mn.FadeIn(bar), mn.FadeIn(number),
                                  lag_ratio=0.3, run_time=2.5),
                mn.AnimationGroup(mn.Transform(bar, bar_disk), *replace(bar_name, bar_disk_name), Count(number, mb_raw, mb_disk, size_label, 2), run_time=2.0),
                mn.FadeIn(pyramid_text, run_time=0.8),
                lag_ratio=1.0),
    ]
    print(f"{n_t} frames at {FS:g} Hz -> {n_t // BIN} at {FS / BIN:g} Hz; {mb_raw:.2f} MB -> {mb_disk:.2f} MB on disk, {len(N['levels'])} pyramid level(s)")
    return timeline([], chapters)


if __name__ == "__main__":
    kit.run("open_scrub_save", CHAPTERS, UNTITLED, build, __doc__)
