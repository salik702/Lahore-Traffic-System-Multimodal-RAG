"""Entry point for the evaluation module."""
from .main import main
import sys
if sys.platform == "win32":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())