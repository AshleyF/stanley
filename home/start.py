"""Run from inside this directory: python start.py"""

import asyncio, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from home.session import main

if __name__ == "__main__":
    asyncio.run(main())
