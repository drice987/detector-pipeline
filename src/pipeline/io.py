from pathlib import Path
from typing import Any

import h5py
import numpy as np

valid_extensions = {".npy", ".h5", ".hdf5", ".sif", ".tif", ".tiff"}


def read_detector_file(
    file_path: Path, config: dict[str, Any]
) -> tuple[np.ndarray, float | None, dict[str, Any]]:
    """Parses various detector formats into a standard 2D array and extracts metadata."""
    i0_val = None
    metadata = {}

    if file_path.suffix == ".sif":
        import sif_parser

        data, _ = sif_parser.np_open(str(file_path))
        raw_2d = np.mean(data, axis=0) if data.shape[0] > 1 else data[0]
        raw_2d = raw_2d.astype(np.float64)

    elif file_path.suffix == ".npy":
        raw_2d = np.load(str(file_path)).astype(np.float64)

    elif file_path.suffix in [".tif", ".tiff"]:
        import tifffile

        raw_2d = tifffile.imread(str(file_path)).astype(np.float64)
        if raw_2d.ndim > 2:
            raw_2d = np.mean(raw_2d, axis=0)

    elif file_path.suffix in [".h5", ".hdf5"]:
        h5_path = config.get("processing", {}).get("h5_data_path", "entry/data/counts")
        i0_path = config.get("processing", {}).get("h5_i0_path", None)

        with h5py.File(file_path, "r") as hf:
            raw_2d = hf[h5_path][:].astype(np.float64)
            if i0_path and i0_path in hf:
                i0_val = float(np.mean(hf[i0_path][()]))

            meta_cfg = config.get("export", {}).get("metadata_paths", "none")
            if meta_cfg != "none":
                if isinstance(meta_cfg, list):
                    for path in meta_cfg:
                        if path in hf:
                            metadata[path] = hf[path][()]
                elif meta_cfg == "all":

                    def safe_visitor(name: str, node: Any) -> None:
                        if isinstance(node, h5py.Dataset) and node.size < 100:
                            metadata[name] = node[()]

                    hf.visititems(safe_visitor)
    else:
        raise ValueError(f"Unsupported file extension: {file_path.suffix}")

    return raw_2d, i0_val, metadata
