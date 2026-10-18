"""The running app is home. This stays so the old command still starts it."""

import asyncio, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from home.session import main

if __name__ == "__main__":
    asyncio.run(main())
