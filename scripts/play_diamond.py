"""Moved to :mod:`simulusion.inference.play_diamond` so the installed CLI can import it.

Kept so ``python scripts/play_diamond.py ...`` keeps working from a checkout.
"""

from simulusion.inference.play_diamond import main

if __name__ == "__main__":
    main()
