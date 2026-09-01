"""Pytest configuration.

Ensures the project root is on ``sys.path`` so ``tests`` can import the
``llm_bot`` package regardless of how pytest is invoked.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)