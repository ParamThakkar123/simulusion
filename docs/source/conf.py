"""Sphinx configuration for Simulusion documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../.."))

project = "Simulusion"
copyright = f"{datetime.now().year}, Param Thakkar"
author = "Param Thakkar"

# Auto-read version from the public simulusion package

import simulusion  # noqa: E402  (must follow the sys.path insertion above)

release = simulusion.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
    "sphinxext.opengraph",
    "sphinx_sitemap",
    "sphinxcontrib.mermaid",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = [
    # Notebook requires pandoc binary; the .md version covers the same content.
    "tutorials/world_model_env_rl_libraries_notebook.ipynb",
]
nbsphinx_execute = "never"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_inherit_docstrings = False
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True
# Render "Attributes:" docstring sections as :ivar: fields inside the class
# description rather than as standalone py:attribute directives. Without this,
# every attribute a class both annotates and documents (each dataclass config,
# for one) is registered twice and Sphinx warns about a duplicate description.
napoleon_use_ivar = True

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "deflist",
    "fieldlist",
]

# Let MyST/Sphinx emit math nodes and let sphinx.ext.mathjax load the
# MathJax runtime exactly once. Keeping MathJax under Sphinx control avoids the
# previous custom-loader race where a configuration object could prevent the
# runtime from loading, leaving equations unrendered.
myst_update_mathjax = False
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "packages": {"[+]": ["ams"]},
        "processEscapes": True,
        "processEnvironments": True,
    },
    "options": {
        "skipHtmlTags": ["script", "noscript", "style", "textarea", "pre", "code"],
    },
}

# Add copy-to-clipboard controls to rendered code blocks. The prompt regexp
# strips common Python and shell prompts when users copy examples.
copybutton_selector = "div.highlight pre"
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_only_copy_prompt_lines = False

# Heavy optional runtimes are mocked so the API reference can build in a docs
# environment without installing all simulation backends.
autodoc_mock_imports = [
    "ale_py",
    "dm_control",
    "deepmind_lab",
    "mlagents_envs",
    "mujoco",
    "moviepy",
    "cv2",
    "pygame",
    "hydra",
    "omegaconf",
    "torchvision",
    "gym",
    "gymnasium",
    "brax",
    "brax.envs",
    "jax",
    "jax.numpy",
    "wandb",
    "PIL",
    "sklearn",
    "sklearn.manifold",
    "h5py",
    "huggingface_hub",
    "matplotlib",
    "matplotlib.pyplot",
    "umap",
    "torch",
    "torch.nn",
    "torch._C",
    "cma",
]

html_theme = "pydata_sphinx_theme"
# Suffix of every page's browser/search-result title ("Dreamer - Simulusion").
# Sphinx's default ("<project> <release> documentation") is long and repeats
# the name; the searchable terms belong in each page's own heading.
html_title = "Simulusion"
html_favicon = "_static/simulusion-favicon.svg"
html_theme_options = {
    "github_url": "https://github.com/paramthakkar123/simulusion",
    "navigation_depth": 3,
    "show_nav_level": 2,
    "logo": {
        "image_light": "simulusion-logo-light.svg",
        "image_dark": "simulusion-logo-dark.svg",
        # The logo image is the wordmark, so no separate text label.
        "alt_text": "Simulusion",
    },
    # Keep the top navbar intentionally minimal: logo, documentation search,
    # the light/dark switch (the docs open in dark mode), and the GitHub link.
    "navbar_start": ["navbar-logo"],
    "navbar_center": [],
    "navbar_end": ["search-field", "theme-switcher", "navbar-icon-links"],
    "navbar_persistent": [],
}

# Open in dark mode. Readers can still switch with the theme toggle, and their
# choice is remembered by the browser.
html_context = {"default_mode": "dark"}

# Include client-side assets. MathJax itself is loaded by sphinx.ext.mathjax;
# adding custom MathJax bootstrap files here can race with Sphinx's runtime and
# prevent equations from being typeset.
html_js_files = [
    # Small script to tidy duplicated navbar elements caused by theme options.
    "fix_navbar.js",
]

# Copy both our _static assets and the `images/` dir so images referenced from
# pages are available at build time (placed under _static/).
html_static_path = ["_static", "images"]

html_css_files = [
    "custom.css",
]

# Canonical docs URL. Overridden in CI for PR previews.
html_baseurl = os.getenv(
    "SPHINX_HTML_BASEURL",
    "https://paramthakkar123.github.io/simulusion/",
)

# Open Graph metadata for social sharing cards/link previews.
ogp_site_url = html_baseurl
ogp_site_name = "Simulusion Documentation"
ogp_description_length = 200
ogp_enable_meta_description = True

# sitemap.xml for search engines. html_baseurl already carries the full docs
# root, so each entry is just the page link (no language/version segments).
sitemap_url_scheme = "{link}"
