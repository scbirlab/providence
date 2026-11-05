from typing import Optional, Iterable, Union

from argparse import Namespace
import os

from carabiner import print_err
from carabiner.cliutils import clicommand


def _validate_checkpoints(chk1, path):
    from ..data import DataResolver
    print_err(f"[INFO] Validating checkpoint at {path}...", end="")
    resolver2 = DataResolver.load_checkpoint(path)
    if chk1 != resolver2:
        attr1, attr2 = vars(chk1), vars(resolver2)
        wrong_attributes = {
            k: (attr1[k], "!=", v) 
            for k, v in attr2.items() 
            if v != attr1[k]
        }
        raise IOError(
            f"Checkpoint at {path} does not recreate the same object. "
            f"These attributes were different: {wrong_attributes}"
        )
    print_err(" all good!")
    return None


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
        _validate_checkpoints(resolved, output)

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
