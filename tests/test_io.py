from pathlib import Path
import sys
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
import tifffile
from pipeline.io import read_detector_file


def test_read_npy(tmp_path):
    arr = np.arange(12, dtype=np.int32).reshape(3, 4)
    path = tmp_path / "scan.npy"
    np.save(path, arr)

    data, i0, meta = read_detector_file(path, {})

    assert data.dtype == np.float64
    assert i0 is None
    assert meta == {}
    np.testing.assert_array_equal(data, arr)


@pytest.mark.parametrize("raw,expected", [
    (np.array([[1, 2], [3, 4]], dtype=np.uint16), [[1.0, 2.0], [3.0, 4.0]]),
    (np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.uint16), [[3.0, 4.0], [5.0, 6.0]]),
])
def test_read_tiff_2d_and_3d(tmp_path, raw, expected):
    path = tmp_path / "scan.tif"
    tifffile.imwrite(path, raw)

    data, i0, _ = read_detector_file(path, {})

    assert data.shape == (2, 2)
    assert i0 is None
    np.testing.assert_allclose(data, expected)


@pytest.mark.parametrize("frames,expected", [
    (np.array([[[2.0, 4.0]]]), [[2.0, 4.0]]),
    (np.array([[[2.0, 4.0]], [[6.0, 8.0]]]), [[4.0, 6.0]]),
])
def test_read_sif_single_and_multi_frame(monkeypatch, frames, expected):
    fake_sif = SimpleNamespace(np_open=lambda _: (frames, {}))
    monkeypatch.setitem(sys.modules, "sif_parser", fake_sif)

    data, _, _ = read_detector_file(Path("run.sif"), {})
    np.testing.assert_allclose(data, expected)


@pytest.mark.parametrize("meta_cfg,expected_keys", [
    ("none", set()),
    (["entry/instrument/temp", "entry/missing_key"], {"entry/instrument/temp"}),
    ("all", {"entry/data/counts", "entry/i0", "entry/instrument/temp"}),
])
def test_read_h5_with_i0_and_metadata(tmp_path, meta_cfg, expected_keys):
    path = tmp_path / "scan.h5"
    counts = np.full((4, 5), 10.0, dtype=np.float32)
    with h5py.File(path, "w") as hf:
        hf.create_dataset("entry/data/counts", data=counts)
        hf.create_dataset("entry/i0", data=np.array([2.0, 4.0]))
        hf.create_dataset("entry/instrument/temp", data=295.15)
        hf.create_dataset("entry/large_array", data=np.zeros(200))

    cfg = {
        "processing": {
            "h5_data_path": "entry/data/counts",
            "h5_i0_path": "entry/i0",
        },
        "export": {"metadata_paths": meta_cfg},
    }

    data, i0, meta = read_detector_file(path, cfg)

    assert i0 == pytest.approx(3.0)
    assert set(meta.keys()) == expected_keys
    np.testing.assert_allclose(data, counts)


def test_unsupported_extension_raises():
    with pytest.raises(ValueError, match="Unsupported file extension"):
        read_detector_file(Path("scan.csv"), {})