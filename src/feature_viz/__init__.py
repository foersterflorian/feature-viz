# feature-viz - live detection with feature-map visualisation
# Copyright (C) 2026  Florian Förster
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Live YOLO26 detection next to the feature maps of selected layers."""

import os

# Telemetry off, for every entry point (DECISIONS.md §19). Whenever ultralytics
# believes it is online it reports usage to Google Analytics, checks PyPI for
# updates and may pip-install missing packages at runtime. YOLO_OFFLINE=1
# switches all of that off; downloading weights still works.
#
# ultralytics reads the variable once, at its first import. That is why it is
# set here: the package __init__ runs before any of its modules can import
# ultralytics. Code that imports ultralytics *before* feature_viz defeats it -
# tests/test_telemetry.py guards the entry points. An explicit YOLO_OFFLINE=0
# in the environment opts back in.
os.environ.setdefault("YOLO_OFFLINE", "1")
