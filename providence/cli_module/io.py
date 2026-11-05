
from carabiner import print_err


def _resolve_and_slice_data(
    data: str,
    start: int | None = None,
    end: int | None = None,
    batch_size: int = 1024,
    cache_dir: str = None
):
    from ..io import AutoDataset

    ds = AutoDataset.load(data, cache_dir)
    nrows = ds._dataset.num_rows
    skip = 0 if start is None else start
    take = (nrows if end is None else end) - skip
    if (take - skip) < nrows:
        print_err(f"[INFO] Reading dataset from row {skip} to row {take + skip} / {nrows}.")
    return ds._dataset.skip(skip).take(take)


def _save_dataset(
    dataset,
    output: str
) -> None:
    print_err("[INFO] Saving dataset:\n" + str(dataset) + "\n" + f"at {output} as", end=" ")
    if output.removesuffix(".gz").endswith((".csv", ".tsv", ".txt")):
        print_err("CSV.")
        dataset.to_csv(
            output, 
            sep="," if output.removesuffix(".gz").endswith(".csv") else "\t",
            compression='gzip' if output.endswith(".gz") else None,
        )
    elif output.endswith(".json"):
        print_err("JSON.")
        dataset.to_json(output)
    elif output.endswith(".parquet"):
        print_err("Parquet.")
        dataset.to_parquet(output)
    elif output.endswith(".sql"):
        print_err("SQL.")
        dataset.to_sql(output)
    elif output.endswith(".hf"):
        print_err("Hugging Face dataset.")
        dataset.save_to_disk(output)
    else:
        print_err("Hugging Face dataset.")
        dataset.save_to_disk(output + ".hf")
        print_err(f"WARNING: Unsure what format to save as for filename {output}. Defaulted to Hugging Face dataset.")
    return None
