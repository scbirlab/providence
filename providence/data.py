"""Data pipeline class."""

from typing import TYPE_CHECKING, Any
from collections.abc import Iterable, Mapping
import os
from pathlib import Path

from carabiner import print_err

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict, IterableDataset
    from .collate import ColumnCollator
else:
    Dataset, DatasetDict, IterableDataset = Any, Any, Any
    ColumnCollator = Any

from . import app_name, __version__
from .io import AutoDataset, DataSource, autoload, file_checksum, save_json, load_json
from .package_data import CACHE_DIR
from .typing import DataLike


DEFAULT_FORMAT: str = "numpy"
CONFIG_FILENAME: str = "config.json"
DATA_FILENAME: str = "data.parquet"
EXAMPLE_FILENAME: str = "example.parquet"


class DataResolver:
    """Data resolving.

    Examples
    ========
    

    """
    def __init__(
        self,
        data: str | Mapping[str, Any],
        start: int | None = None,
        end: int | None = None,
        cache_dir: str | None = None,
        _file_checksum: str | None = None,
        _version: str = __version__,
        _app: str = app_name
    ):
        self._version = _version
        self._app = _app
        self._data_requested = data
        self.data = None
        self._file_checksum = _file_checksum
        self.slice = (start, end)
        self.data_source = None
        self.data_example = None
        self.cache_dir = cache_dir or CACHE_DIR
        self._data_loaded = False

        self.data = self._resolve_data(
            self._data_requested,
            *self.slice,
        )
        if self.data is None:
            raise AttributeError(
                f"Data could not be loaded from {self._data_requested}."
            )
        self._data_loaded = self.data is not None
        self.data_example, self.data_shape = self._inspect_data(
            self.data,
            self.data_example,
        )
    
    def _resolve_data(
        self,
        data: DataLike,
        start: int | None = None,
        end: int | None = None,
        cache_dir: str | None = None
    ) -> Dataset | IterableDataset:
        resolved = AutoDataset.load(
            data, 
            cache_dir=cache_dir or self.cache_dir,
        )
        nrows = resolved._dataset.num_rows
        skip = 0 if start is None else start
        take = (nrows if end is None else end) - skip
        if (take - skip) < nrows:
            print_err(
                f"[INFO] Reading dataset from row {skip} to row "
                f"{take + skip} / {nrows}."
            )
        self.data_source = resolved.source
        return resolved._dataset.skip(skip).take(take)

    @staticmethod
    def _inspect_data(data: Dataset, example=None, format: str = DEFAULT_FORMAT) -> None:
        if example is None:
            if data is not None:
                data = data.with_format(format)
                example = data.take(1)
            else:
                raise AttributeError("Cannot inspect data without loading data or having an example")
        
        first_item = example.with_format(format)[:1]
        shape = {
            col: first_item[col].shape[1:]
            if not isinstance(first_item[col], dict)
            else {
                k: v.shape[1:] if v is not None else None 
                for k, v in first_item[col].items()
            }
            for col in example.column_names
        }
        return example, shape

    def save(
        self,
        path: str | os.PathLike,
        *,
        save_source_data: bool | None = None,
        discard_example_data: bool = False
    ) -> None:
        """Save the pipeline and required training-data artefacts.

        Parameters
        ==========
        path
            Checkpoint directory.
        save_source_data
            Whether to embed the input dataset.

            ``None`` chooses automatically: immutable remote sources are
            referenced, while local or ephemeral sources are packaged.

            ``True`` always packages the resolved input data.

            ``False`` never packages it.
        discard_example_data
            Normally, save one processed example when available. Set 
            discard_example_data=True to turn off.

        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.data_source is None:
            source = DataSource()
        else:
            source = self.data_source

        if save_source_data is None:
            save_source_data = not source.is_remote
        if save_source_data:
            data_filename = str(path / DATA_FILENAME)
            (
                self.data
                .with_format(None)
                .to_parquet(data_filename)
            )
            checksum = file_checksum(data_filename)
        else:
            checksum = None

        checkpoint_config = {
            "source": source.to_config(),
            "slice": list(self.slice),
            "artefacts": {
                "source_data": DATA_FILENAME if save_source_data else None,
                "data_checksum": checksum,
                "example": (
                    EXAMPLE_FILENAME
                    if (
                        not discard_example_data
                        and self.data_example is not None
                    )
                    else None
                ),
            },
        }

        save_json(
            checkpoint_config, 
            str(path / CONFIG_FILENAME),
        )

        if (
            not discard_example_data
            and self.data_example is not None
        ):
            (
                self.data_example
                .with_format(None)
                .to_parquet(str(path / EXAMPLE_FILENAME))
            )

    @classmethod
    def load(
        cls,
        path: str | os.PathLike,
        *,
        cache_dir: str | None = None,
    ) -> "DataResolver":
        """Restore a saved DataResolver and its available data artefacts."""
        path = Path(path)
        config = load_json(path / CONFIG_FILENAME)

        data_source = DataSource.from_config(
            config.get("source")
        )
        artefacts = config.get("artefacts", {})
        data_filename = artefacts.get("source_data")
        data_checksum = artefacts.get("data_checksum") 
        example_filename = artefacts.get("example")
        _slice = config.get("slice", (None, None))

        if data_filename is not None:
            data_uri = str(path / data_filename)
            this_checksum = file_checksum(data_uri)
            if not this_checksum == data_checksum:
                raise IOError(f"Checksum for {data_uri} should be {data_checksum}, but was {this_checksum}.")
            start, end = None, None
        elif data_source is not None:
            if (
                data_source.uri is not None 
                and not data_source.is_remote
                and not os.path.exists(data_source.uri)
            ):
                raise FileNotFoundError(
                    "Checkpoint references external training data "
                    f"at {data_source.uri}, but that source does not exist."
                )
            data_source.assert_verified()
            data_uri = data_source.uri
            start, end = _slice
        else:
            raise AttributeError(f"Checkpoint is missing a data definition!\n{config=}")

        data = cls(
            data=data_uri,
            start=start,
            end=end,
            cache_dir=cache_dir,
            _file_checksum=data_checksum,
        )
        if data_source is not None:
            data.data_source = data_source

        if example_filename is not None:
            data.data_example = autoload(
                path / example_filename,
                cache_dir=cache_dir,
            )
        if data.data_example is not None:
            data.data_example, data.data_shape = data._inspect_data(
                data.data, 
                data.data_example,
            )
        return data

    # for compatibility
    def save_checkpoint(self, *args, **kwargs):
        return self.save(*args, **kwargs)

    @classmethod
    def load_checkpoint(cls, *args, **kwargs):
        return cls.load(*args, **kwargs)
