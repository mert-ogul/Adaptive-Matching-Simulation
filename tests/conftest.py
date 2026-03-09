"""Pytest configuration: ensure src is on the path so tests can import the package."""

import sys
from pathlib import Path

# Add project src so "import expert_sim" works when run from repo root
# (e.g. pytest without pip install -e . or when Python version differs from install)
_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
