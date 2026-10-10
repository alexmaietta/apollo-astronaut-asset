"""Warp kernels and GPU helpers for the Newton examples: full-body IK, hand control, haptics and skinning.

The rig, pose library and hand model they build on live in the repo's scripts/ (plain numpy / OpenUSD, shared
with the asset build), which this package puts on sys.path.
"""
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_SCRIPTS = os.path.join(REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
