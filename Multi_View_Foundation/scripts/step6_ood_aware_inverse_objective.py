#!/usr/bin/env python
"""Backward-compatible wrapper for the redesigned F6 interpretability step."""

from step6_dit_interpretability import build_arg_parser, main


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
