from argparse import Namespace
import os

from carabiner.cliutils import clicommand


@clicommand("Fetching data with the following parameters")
def _write(args: Namespace) -> None:

    from ..data import DataResolver
    
    resolved = DataResolver.load(
        args.input_file,
        cache_dir=args.cache,
    )

    if args.output:
        resolved.write_table(
            args.output,
            include_source=args.with_info,
        )
    return None
