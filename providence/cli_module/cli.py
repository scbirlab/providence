"""Command-line interface for providence."""

from argparse import FileType
import os
import sys

from carabiner.cliutils import CLIOption, CLICommand, CLIApp

from .. import app_name, __version__
from .fetch import _fetch

def main() -> None:

    input_file = CLIOption(
        'input_file',
        type=FileType('r'),
        default=sys.stdin,
        nargs='?',
        help='Input file. Default: STDIN',
    )
    input_filename = CLIOption(
        'input_file',
        type=str,
        help='Input file.',
    )
    force_write = CLIOption(
        '--force-write-data', '-f',
        action='store_true',
        help='Save data to disk from remote repos, instead of simply saving a reference.',
    )
    cache = CLIOption(
        '--cache',
        type=str,
        default=None,
        help='Where to cache data.',
    )
    output_name = CLIOption(
        '--output', '-o', 
        type=str,
        default=False,
        help='Output filename for data table.',
    )
    checkpoint = CLIOption(
        '--checkpoint', '-c', 
        type=str,
        default=False,
        help='Output filename for Providence checkpoint.',
    )

    # slice dataset
    slice_start = CLIOption(
        '--start', 
        type=int,
        default=None,
        help='First row of dataset to process.',
    )
    slice_end = CLIOption(
        '--end', 
        type=int,
        default=None,
        help='Last row of dataset to process. Default: end of dataset.',
    )

    get_command = CLICommand(
        "get",
        description="Load a local or remote dataset, and save as a reloadable and verifiable providence checkpoint.",
        options=[
            input_filename,
            slice_start,
            slice_end,
            force_write,
            checkpoint,
            output_name,
            cache,
        ],
        main=_fetch,
    )
    
    app = CLIApp(
        app_name, 
        description="Data ingestion, validation, and provenance.", 
        version=__version__, 
        commands=[
            get_command,
        ],
    )
    app.run()
    return None


if __name__ == '__main__':
    main()
