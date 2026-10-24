#!/bin/bash
# Refresh the hard-state archive every 10 minutes (atomic replace inside build_archive.py).
cd /d/dev/CCC-OMNI/workspace/colony-one-level4
while [ ! -f STOP_archive ]; do ../../.venv/Scripts/python build_archive.py >> logs/archive.txt 2>&1; sleep 600; done
