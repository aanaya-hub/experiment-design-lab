"""
conftest.py — make `src/` importable from the tests.

WHY THIS FILE EXISTS
--------------------
The modules live in `src/` and the tests live in `tests/`. Without this, `import panel` fails from a
test and the fix people reach for — copying code into the test — is exactly what tests exist to
prevent. This adds the directory to the import path once, for every test.
"""

# sys: the import path is a list, and this appends to it.
import sys
# pathlib: resolve paths relative to this file, not to the working directory.
from pathlib import Path

# __file__ is tests/conftest.py -> .parent is tests/ -> .parent.parent is the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Put the project root on the path so `from src.panel import ...` resolves.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
