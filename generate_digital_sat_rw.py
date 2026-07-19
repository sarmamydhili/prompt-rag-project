#!/usr/bin/env python3
"""Entry point for Digital SAT Reading and Writing question generation."""

import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from digital_sat_generation.cli import main

if __name__ == "__main__":
    main()
