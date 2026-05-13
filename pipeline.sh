#!/bin/bash
#
uv run prepare.py
# Run profile
uv run profile.py --model models/pi05_openpi.py --class-name PI05AutoKernelModel   --input-shape 1,10 --dtype bfloat16

uv run extract.py --top 10