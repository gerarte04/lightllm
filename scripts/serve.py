#!/usr/bin/env python3
"""Launch the OpenAI-compatible API server. See core/api.py for options."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.api import main

if __name__ == "__main__":
    main()
