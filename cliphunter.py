#!/usr/bin/env python
"""ClipHunter entry point.

Usage:
    python cliphunter.py "https://twitch.tv/videos/123456789" --clips 10
"""

from __future__ import annotations

from cliphunter.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
