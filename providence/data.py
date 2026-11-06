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
        self.selected_rows = None
        self.data_source = None
        self.data_example = None
        self.cache_dir = cache_dir or CACHE_DIR
        self._data_loaded = False
        self._init()
        

    def _init(self) -> None:
        self.data = self._resolve_data(
            self._data_requested,
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
        return None

    def _resolve_data(
        self,
        data: DataLike,
        cache_dir: str | None = None
    ) -> Dataset | IterableDataset:
        resolved = AutoDataset.load(
            data, 
            cache_dir=cache_dir or self.cache_dir,
        )
        self.data_source = resolved.source
        return resolved._dataset

    def _apply_selection(
        self,
        start: int | None = None,
        end: int | None = None
    ) -> "DataResolver":
        if self.selected_rows is None:
            selected_rows = []
        else:
            selected_rows = self.selected_rows.copy()
        
        nrows = self.data.num_rows
        indices = slice(start, end).indices(nrows)
        skip = indices[0]
        take = indices[1] - skip

        selected = type(self)(
            self.data.skip(skip).take(take),
            cache_dir=self.cache_dir,
        )

        selected.data_source = self.data_source
        selected_rows.append((start, end))
        selected.selected_rows = selected_rows
        return selected

    @staticmethod
    def _normalize_selected_rows(
        start: Iterable[Iterable[int]] | int | None = None,
        end: int | None = None
    ):
        if start is None and end is None:
            return None
        if isinstance(start, int) or (start is None and isinstance(end, int)):
            return [(start, end)]
        elif isinstance(start, Iterable) and not isinstance(start, str):
            start = list(start)
            if len(start) == 0:
                raise ValueError(f"start was iterable but length-0: {start}")
            elif len(start) == 2 and all(isinstance(item, int) for item in start):
                start = [start]
            for i, item in enumerate(start):
                if isinstance(item, Iterable) and not isinstance(item, str):
                    if len(item) != 2:
                        raise ValueError(f"start was iterable but item {i} was not a 2-tuple: {item}")
                else:
                    raise ValueError(f"start was iterable but item {i} was not: {item}\n{start=}")
            return [tuple(item) for item in start]
        else: 
            raise ValueError(f"Cannot normalize inputs: {start=}, {end=}")
        
    def select_rows(
        self,
        start: Iterable[Iterable[int]] | int | None = None,
        end: int | None = None
    ) -> "DataResolver":
        """Return a new resolver containing rows [start:end]."""

        if start is None and end is None:
            return self  # pass through
        selections = self._normalize_selected_rows(start, end)
        data = self
        for item in selections:
            data = data._apply_selection(*item)
        return data

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
            "selected_rows": self.selected_rows,
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
        selected_rows = config.get("selected_rows")

        if data_filename is not None:
            data_uri = str(path / data_filename)
            this_checksum = file_checksum(data_uri)
            if not this_checksum == data_checksum:
                raise IOError(f"Checksum for {data_uri} should be {data_checksum}, but was {this_checksum}.")
            has_embedded_data = True
        elif data_source is not None:
            if data_source.uri is None:
                raise ValueError(
                    f"Reference-only checkpoint has no source URI:\n{data_source}"
                )
            elif (
                not data_source.is_remote
                and not os.path.exists(data_source.uri)
            ):
                raise FileNotFoundError(
                    "Checkpoint references non-remote data "
                    f"at {data_source.uri}, but that source does not exist."
                )
            data_source.assert_verified()
            data_uri = data_source.uri
            has_embedded_data = False
        else:
            raise AttributeError(f"Checkpoint is missing a data definition!\n{config=}")

        data = cls(
            data=data_uri,
            cache_dir=cache_dir,
        )
        if not has_embedded_data and selected_rows is not None:
            data = data.select_rows(*selected_rows)
        data.selected_rows = cls._normalize_selected_rows(selected_rows)
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
