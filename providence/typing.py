"""Custom types."""

from typing import TYPE_CHECKING, Any, TypeAlias, Union
from collections.abc import Iterable, Mapping
from os import PathLike

from datasets import Dataset, IterableDataset
from pandas import DataFrame
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from .transform.base import ColumnTransform
else:
    ColumnTransform = Any

DataLike: TypeAlias = Union[
    str, 
    DataFrame, 
    Mapping[str, ArrayLike], 
    Dataset, 
    IterableDataset,
    PathLike
]

StrOrIterableOfStr = Union[str, Iterable[str]]

Datum: TypeAlias = Mapping[str, Any]
Batch: TypeAlias = Iterable[Mapping[str, Any]]
