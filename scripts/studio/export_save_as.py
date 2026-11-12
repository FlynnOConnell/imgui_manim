"""
The numbers the open-scrub-save animation shows, from mbo_utilities itself: the demo movie's metadata as the Studio
reads it, the frame-averaged view, and an OME-Zarr written with the Save As dialog's zarr options (sharded, OME,
gzip level 1, pyramid up to 3 levels). Writes animations/data/open_scrub_save/save_as.json.

    python scripts/studio/export_save_as.py
"""

import json
import shutil
import tempfile
from pathlib import Path

import mbo_utilities as mbo

FILEPATH_TIF = Path.home() / "data" / "demoMovie.tif"
FS_HZ = 10.0  ## CaImAn's params_demo_OnACID.json: "fr": 10; the tif itself carries no frame interval
FRAME_AVERAGE = 10
COMPRESSION_LEVEL = 1
PYRAMID_MAX_LAYERS = 3
DIR_OUT = Path(__file__).resolve().parents[2] / "animations" / "data" / "open_scrub_save"


def codec_names(codecs: list) -> list:
    out = []
    for codec in codecs:
        out.append(codec["name"])
        if codec["name"] == "sharding_indexed":
            out += codec_names(codec["configuration"]["codecs"])
    return out


def level_info(dir_store: Path) -> list:
    levels = []
    for filepath_meta in sorted(dir_store.rglob("zarr.json")):
        meta = json.loads(filepath_meta.read_text())
        if meta.get("node_type") != "array":
            continue
        dir_level = filepath_meta.parent
        codecs = meta["codecs"]
        inner = next((c["configuration"]["chunk_shape"] for c in codecs if c["name"] == "sharding_indexed"), None)
        levels.append({
            "path": dir_level.name,
            "shape": meta["shape"],
            "dtype": meta["data_type"],
            "shard_shape": meta["chunk_grid"]["configuration"]["chunk_shape"],
            "chunk_shape": inner,
            "codecs": codec_names(codecs),
            "bytes_on_disk": sum(f.stat().st_size for f in dir_level.rglob("*") if f.is_file()),
        })
    return levels


def main() -> None:
    raw = mbo.imread(str(FILEPATH_TIF))
    binned = mbo.imread(str(FILEPATH_TIF), frame_average=FRAME_AVERAGE)
    dir_tmp = Path(tempfile.mkdtemp(prefix="open_scrub_save_"))
    filepath_store = mbo.imwrite(
        binned, str(dir_tmp / "demoMovie.zarr"), ext=".zarr", overwrite=True, sharded=True, ome=True,
        compressor="gzip", compression_level=COMPRESSION_LEVEL, pyramid=True, pyramid_max_layers=PYRAMID_MAX_LAYERS,
        metadata={"fs": FS_HZ / FRAME_AVERAGE},
    )
    store = Path(filepath_store)
    attrs = json.loads((store / "zarr.json").read_text())["attributes"]
    levels = level_info(store)
    shape_binned = [int(s) for s in binned.shape]
    out = {
        "filepath_tif": FILEPATH_TIF.name,
        "fs_hz": FS_HZ,
        "frame_average": FRAME_AVERAGE,
        "raw": {"shape": [int(s) for s in raw.shape], "dtype": str(raw.dtype), "fs": raw.metadata.get("fs"),
                "num_timepoints": int(raw.shape[0]), "value_range": [int(raw[:, 0, 0].min()), int(raw[:, 0, 0].max())]},
        "binned": {"shape": shape_binned, "num_timepoints": shape_binned[0], "fs": FS_HZ / FRAME_AVERAGE},
        "store_name": store.name,
        "store_parent": store.parent.name,
        "ome_version": attrs["ome"]["version"],
        "axes": [a["name"] for a in attrs["ome"]["multiscales"][0]["axes"]],
        "pyramid_method": attrs["ome"]["multiscales"][0]["metadata"]["method"],
        "compression_level": COMPRESSION_LEVEL,
        "pyramid_max_layers": PYRAMID_MAX_LAYERS,
        "pyramid_min_size_px": 64,
        "levels": levels,
        "bytes_uncompressed": int(2 * shape_binned[0] * shape_binned[-2] * shape_binned[-1]),
    }
    DIR_OUT.mkdir(parents=True, exist_ok=True)
    (DIR_OUT / "save_as.json").write_text(json.dumps(out, indent=1))
    shutil.rmtree(dir_tmp)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
