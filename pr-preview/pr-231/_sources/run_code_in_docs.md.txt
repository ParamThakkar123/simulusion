# Run Code In The Docs

This project enables running Python examples directly in the documentation using Thebe + Binder.

How it works
------------

- The documentation includes Thebe (a small JS client) which can request a live Jupyter kernel from Binder.
- When you open a page with a code block, click the run button that appears to execute the cell in a live kernel and see outputs inline.

Binder environment
------------------

- A minimal Binder environment is provided at `binder/requirements.txt` for lightweight examples.
- Heavy libraries (PyTorch) are intentionally not included by default because they make Binder builds very slow and often fail due to size limits. If you need PyTorch in Binder, create a custom Binder environment with CPU-only wheels and update the `thebe_config` accordingly.

Usage notes & caveats
---------------------

1. Click the run/play button in the top-right of a code block to run it.
2. First-run may take time while Binder spins up a server (a few tens of seconds to minutes).
3. Sessions are temporary. Save work to your local machine if needed.

Security
--------

Running code on Binder executes on remote compute provided by Binder. Be cautious with secrets or downloading large datasets.

Customization
-------------

To point Thebe at a different Binder repo/branch, edit `thebe_config` in `docs/source/conf.py` and rebuild the docs.
