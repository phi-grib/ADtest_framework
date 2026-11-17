"""
Main entry point for the ADTest command-line interface.

This script provides a flexible CLI that can be driven either by a YAML
configuration file (for complex, reproducible runs) or by direct command-line
arguments (for quick, exploratory analyses).

Usage examples
──────────────
  # YAML-driven batch run (recommended for publications):
  adtest -c run -cfg my_config.yaml

Author: Eric March Vila (eric.march@upf.edu)
Created: 2025-10-10
Refactored: 2026-08-10
"""

import argparse
import sys

from . import context as ctx

def main():
    parser = argparse.ArgumentParser(
        prog='adtest',
        description=(
            'ADTest: Applicability Domain analysis for QSAR and '
            'in-silico risk assessment.\n\n'
            'Accepts a YAML configuration file (-cfg) for batch/reproducible runs'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
────────
  # YAML-driven batch run:
  adtest -c run -cfg my_config.yaml
        """
    )

    # ── Top-level command ─────────────────────
    parser.add_argument(
        '-c', '--command',
        action='store',
        choices=['run'],
        required=True,
        metavar='CMD',
        help=(
            "Command to execute. Choices: "
            "'run' (AD analysis pipeline)"
        )
    )

    # ── Input: YAML config ────────────
    parser.add_argument(
        '-cfg', '--config',
        metavar='FILE',
        help=(
            "Path to a YAML configuration file. When provided, all other "
            "data/model flags below are ignored (the YAML takes precedence). "
            "Use this for reproducible, publication-quality runs."
        )
    )

    parser.add_argument(
        '-o', '--output',
        metavar='DIR',
        default='results',
        help=(
            "Base output directory for all results, logs, plots, and reports. "
            "Each endpoint gets its own subdirectory: <output>/<endpoint>/. "
            "(default: 'results/')"
        )
    )

    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help="Enable quiet mode. Suppresses standard logging output."
    )


    args = parser.parse_args()

    # Create the application context and execute the selected command
    context = ctx.ADTestContext(args)
    context.execute()


if __name__ == '__main__':
    main()