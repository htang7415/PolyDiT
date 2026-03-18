#!/usr/bin/env python
"""Backward-compatible wrapper for the renamed F4 embedding research step."""

from step4_embedding_research import build_parser, main


if __name__ == "__main__":
    main(build_parser().parse_args())
