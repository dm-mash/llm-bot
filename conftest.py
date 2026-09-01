"""Pytest configuration.

- Ensures the project root is on ``sys.path`` so ``tests`` can import the
  ``llm_bot`` package regardless of how pytest is invoked.
- Loads ``.env.test`` (overriding any existing values) *before* anything else,
  so a real ``.env`` with actual credentials never leaks into the test run.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Must run before llm_bot.config is imported: with override=True these values
# land in os.environ first, and config.py's own load_dotenv() (override=False)
# will not clobber them.
load_dotenv(ROOT / ".env.test", override=True)