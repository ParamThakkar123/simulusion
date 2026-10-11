"""Moved to :mod:`simulusion.inference.eval_diamond` so the installed CLI can import it.

Kept so ``python scripts/eval_diamond.py ...`` keeps working from a checkout.
"""

from simulusion.inference.eval_diamond import main

if __name__ == "__main__":
    main()
