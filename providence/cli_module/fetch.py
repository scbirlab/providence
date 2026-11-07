from argparse import Namespace
import os

from carabiner.cliutils import clicommand


@clicommand("Fetching data with the following parameters")
def _fetch(args: Namespace) -> None:

    from .io import _save_dataset
    from ..data import DataResolver
    
    resolved = DataResolver(
        args.input_file,
        cache_dir=args.cache,
    )
    if args.start is not None or args.end is not None:
        resolved = resolved.select_rows(
            args.start,
            args.end,
        )
    
    if not args.checkpoint and not args.output:
        raise ValueError("At least one of --checkpoint and --output should be set.")
    
    if args.checkpoint:
        output = args.checkpoint
        out_dir = os.path.dirname(output)
        base = os.path.basename(output)
        
        resolved.save(output, save_source_data=args.force_write_data)

    if args.output:
        resolved.write_table(
            args.output,
            include_source=args.with_info,
        )
    return None
