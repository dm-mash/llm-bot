"""Allow running the console app via ``python -m llm_bot``."""

from llm_bot.cli import main

if __name__ == "__main__":
    raise SystemExit(main())