"""Make the package importable when it has not been pip-installed.

With ``pip install -e .`` this does nothing. Without it, a fresh clone can still
run the tests.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
