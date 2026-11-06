from argparse import Namespace
import os

from carabiner.cliutils import clicommand


@clicommand("Fetching data with the following parameters")
def _fetch(args: Namespace) -> None:

    from .io import _save_dataset
    from ..data import DataResolver
    
    resolved = DataResolver(
        args.input_file,
        start=args.start,
        end=args.end,
        cache_dir=args.cache,
    )
    
    if not args.checkpoint and not args.output:
        raise ValueError("At least one of --checkpoint and --output should be set.")
    
    if args.checkpoint:
        output = args.checkpoint
        out_dir = os.path.dirname(output)
        base = os.path.basename(output)
        
        resolved.save_checkpoint(output, save_source_data=args.force_write_data)

    if args.output:
        output = args.output
        out_dir = os.path.dirname(output)
        base = os.path.basename(output)
        if len(out_dir) > 0:
            os.makedirs(out_dir, exist_ok=True)
        _save_dataset(
            resolved.data, 
            output,
        )
    return None
