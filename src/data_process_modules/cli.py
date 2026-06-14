"""Compatibility CLI for the public data_process_modules package."""
from __future__ import annotations

import json
import sys

from dataops.cli import build_parser, main as _dataops_main

from .registry import MANIFEST


def main(argv=None) -> None:
    args = sys.argv[1:] if argv is None else list(argv)
    if args and args[0] == "manifest":
        json.dump(MANIFEST, sys.stdout, indent=2, default=str)
        sys.stdout.write("\n")
        return
    _dataops_main(args)

__all__ = ["build_parser", "main"]
