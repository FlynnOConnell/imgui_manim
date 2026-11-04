"""Write the suite2p results the animation draws (mean image, cell contours, dF/F traces, step timings) as one npz."""

import json
import sys
from pathlib import Path

import numpy as np
from skimage import measure

from lbm_suite2p_python import dff_rolling_percentile

dir_plane = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "data" / "demoMovie_studio_demo" / "zplane01_tp00001-02000"
filepath_out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "animations" / "data" / "suite2p_demo" / "suite2p_demo.npz"
filepath_out.parent.mkdir(parents=True, exist_ok=True)

ops = np.load(dir_plane / "ops.npy", allow_pickle=True).item()
stat = np.load(dir_plane / "stat.npy", allow_pickle=True)
iscell = np.load(dir_plane / "iscell.npy")[:, 0].astype(bool)
F = np.load(dir_plane / "F.npy")
Fneu = np.load(dir_plane / "Fneu.npy")
spks = np.load(dir_plane / "spks.npy")
ly, lx = int(ops["Ly"]), int(ops["Lx"])

contours = []
for s in stat:
    mask = np.zeros((ly + 2, lx + 2), dtype=bool)
    mask[s["ypix"] + 1, s["xpix"] + 1] = True
    polygon = max(measure.find_contours(mask.astype(float), 0.5), key=len) - 1.0  ## (n, 2) rows, cols; pixel centers at integers
    contours.append(polygon.astype(np.float32))

## NOTE: raw F, no neuropil subtraction: F - 0.7 Fneu crosses zero on this movie and dF/F blows up
dff = dff_rolling_percentile(F, window_size=int(60 * ops["fs"]), percentile=20)
np.savez(
    filepath_out,
    mean_img=ops["meanImg"].astype(np.float32),
    max_proj=ops["max_proj"].astype(np.float32),
    iscell=iscell,
    contours=np.array(contours, dtype=object),
    dff=dff.astype(np.float32),
    spks=spks.astype(np.float32),
    fs=float(ops["fs"]),
    timings=json.dumps(ops["plane_times"]),
)
print(filepath_out, "cells", int(iscell.sum()), "of", len(stat), "dff range", float(dff[iscell].min()), float(dff[iscell].max()))
print("accepted cells by peak dF/F:", np.argsort(-dff[iscell].max(axis=1))[:8], np.flatnonzero(iscell)[np.argsort(-dff[iscell].max(axis=1))[:8]])
