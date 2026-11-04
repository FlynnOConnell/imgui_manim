"""Run suite2p on demoMovie.tif the way the Studio's Run tab does (lbm_suite2p_python.run_plane, fs 10 Hz, tau 1 s, diameter 12 px, rigid registration: nonrigid 128 px blocks crash on a 60 x 80 px frame)."""

import sys
from pathlib import Path

import mbo_utilities as mbo
from lbm_suite2p_python import run_plane

filepath_movie = Path.home() / "data" / "demoMovie.tif"
dir_out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "data" / "demoMovie_studio_demo"
array = mbo.imread(str(filepath_movie))
array.metadata["fs"] = 10.0
plane_dir = run_plane(array, save_path=str(dir_out), ops={"fs": 10.0, "tau": 1.0, "diameter": [12.0, 12.0], "nonrigid": False}, plane_name="zplane01_tp00001-02000")
print("plane dir", plane_dir)
