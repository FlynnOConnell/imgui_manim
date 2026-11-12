# /// script
# requires-python = ">=3.11"
# dependencies = ["cloud-volume", "numpy"]
# ///
"""
Downloads the MICrONS pyramidal cell's static mesh (public bucket, no account) to ~/data/microns_pyramidal, 240 MB.

    uv run -p 3.12 scripts/synapse_to_soma/fetch_microns_cell.py
"""

from pathlib import Path

import numpy as np
from cloudvolume import CloudVolume

CELL = 864691135572530981  # the pyramidal cell in the MICrONS tutorials, segmentation v1300
dir_mesh = Path.home() / "data/microns_pyramidal"
dir_mesh.mkdir(parents=True, exist_ok=True)
volume = CloudVolume("precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300", use_https=True, progress=False)
for lod in (1, 0):
    mesh = volume.mesh.get(CELL, lod=lod)[CELL]
    np.savez(dir_mesh / f"cell_lod{lod}.npz", vertices=np.asarray(mesh.vertices, np.float32), faces=np.asarray(mesh.faces, np.int32))
