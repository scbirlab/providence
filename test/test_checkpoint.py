import json

import pandas as pd
import pytest

from providence import DataResolver, DataSource


DATA = {
    "x": [1., 2., 4.],
    "labels": [0, 1, 0],
}


def make_data(tmp_path):
    data = DataResolver(
        DATA,
        cache_dir=tmp_path / "cache",
    )
    return data


def test_ephemeral_source_is_packaged(tmp_path):
    data = make_data(tmp_path)

    checkpoint = tmp_path / "checkpoint"

    data.save(checkpoint)

    assert (checkpoint / "config.json").exists()
    assert (checkpoint / "data.parquet").exists()
    assert not (checkpoint / "transformed.parquet").exists()

    assert (checkpoint / "example.parquet").exists()

    restored = DataResolver.load(
        checkpoint,
        cache_dir=tmp_path / "load-cache",
    )

    assert restored.data is not None
    assert len(restored.data) == 3


def test_force_no_source_packaging(tmp_path):
    data = make_data(tmp_path)

    checkpoint = tmp_path / "checkpoint"

    data.save(
        checkpoint,
        save_source_data=False,
    )

    assert not (checkpoint / "data.parquet").exists()

    with pytest.raises(
        ValueError,
        match="no source URI",
    ):
        restored = DataResolver.load(
            checkpoint,
            cache_dir=tmp_path / "load-cache",
        )


def test_remote_source_is_reference_only(tmp_path):
    
    data = make_data(tmp_path)
    data.data_source = DataSource(
        uri="hf://datasets/example/data@abc123:train",
        requested_uri="hf://datasets/example/data@main:train",
        revision="abc123",
        requested_revision="main",
    )

    checkpoint = tmp_path / "checkpoint"
    data.save(checkpoint)

    assert not (checkpoint / "data.parquet").exists()

    with open(checkpoint / "config.json") as file:
        config = json.load(file)

    assert config["source"] == {
        "checksum": None,
        "uri": "hf://datasets/example/data@abc123:train",
        "requested_uri": "hf://datasets/example/data@main:train",
        "revision": "abc123",
        "requested_revision": "main",
    }


def test_checkpoint_reconstructs_verified_local_source(tmp_path):

    source = tmp_path / "train.parquet"

    pd.DataFrame({
        "x": [1., 2., 3.],
        "y": [2., 4., 6.],
    }).to_parquet(
        source,
        index=False,
    )

    data = DataResolver(source)

    checkpoint = tmp_path / "data"

    data.save(
        checkpoint,
        save_source_data=False,
    )

    restored = DataResolver.load(checkpoint)

    assert restored.data is not None
    assert restored.data_source.verify() is True


def test_checkpoint_rejects_modified_local_source(tmp_path):

    source = tmp_path / "train.parquet"

    pd.DataFrame({
        "x": [1., 2., 3.],
    }).to_parquet(
        source,
        index=False,
    )

    data = DataResolver(source)

    checkpoint = tmp_path / "data"

    data.save(
        checkpoint,
        save_source_data=False,
    )

    pd.DataFrame({
        "x": [1., 2., 4.],
    }).to_parquet(
        source,
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="checksum",
    ):
        DataResolver.load(checkpoint)


def test_embedded_checkpoint_ignores_modified_original_source(
    tmp_path,
):

    source = tmp_path / "train.parquet"

    pd.DataFrame({
        "x": [1., 2., 3.],
    }).to_parquet(
        source,
        index=False,
    )

    data = DataResolver(str(source))

    checkpoint = tmp_path / "data"

    data.save(
        checkpoint,
        save_source_data=True,
    )

    pd.DataFrame({
        "x": [9., 9., 9.],
    }).to_parquet(
        source,
        index=False,
    )

    restored = DataResolver.load(checkpoint)

    assert restored.data["x"] == [1., 2., 3.]


def test_embedded_slice_roundtrip(tmp_path):
    original = DataResolver(
        {"x": list(range(10))},
        cache_dir=str(tmp_path / "cache"),
    )

    selected = original.select_rows(2, 5)

    assert selected.data["x"] == [2, 3, 4]
    assert original.data["x"] == list(range(10))
    assert selected.data_source == original.data_source

    path = tmp_path / "checkpoint"
    selected.save(path)

    restored = DataResolver.load(path)

    assert restored.data["x"] == [2, 3, 4]
    assert restored.selected_rows == selected.selected_rows


def test_reference_slice_roundtrip(tmp_path):
    source = tmp_path / "data.parquet"

    pd.DataFrame({
        "x": list(range(10)),
    }).to_parquet(source, index=False)

    selected = (
        DataResolver(source)
        .select_rows(2, 5)
    )

    path = tmp_path / "checkpoint"
    selected.save(path, save_source_data=False)

    assert not (path / "data.parquet").exists()

    restored = DataResolver.load(path)

    assert restored.data["x"] == [2, 3, 4]
    assert restored.selected_rows == selected.selected_rows
    assert restored.data_source.verify() is True
