"""Run a module's own ``app.py`` as a page of the unified shell.

Each of the three Streamlit modules (chargeback, account_takeover, phishing)
was written as a standalone top-level app: it expects its own directory to be
``sys.path[0]`` and it calls ``st.set_page_config()``. Rather than editing
those files -- which would fork them from the branches their authors still
work on -- this loader reproduces the environment they expect and executes
them verbatim.

Two things have to be arranged:

1. ``sys.path``. ``streamlit run <dir>/app.py`` puts ``<dir>`` at the front of
   ``sys.path``; ``st.Page`` does not, because the shell's own directory is
   the script root. Without this, ``account_takeover``'s ``import config as
   cfg`` fails outright.

2. ``st.set_page_config``. Streamlit permits exactly one call per process, and
   the shell's ``app.py`` has already made it. The module's call is
   neutralised for the duration of its run and then restored.

Top-level module names the hosted modules publish on ``sys.path`` -- do not
create a root-level module with any of these names, it would shadow theirs:

    account_takeover -> config
    phishing         -> schema, personas, templates, llm_client,
                        dataset_utils, train_baseline, adversarial_loop

They do not collide with each other today. ``chargeback`` is safe either way:
everything it owns lives under the ``aegis`` package.
"""

from __future__ import annotations

import runpy
import sys
from contextlib import contextmanager
from pathlib import Path

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULES = REPO_ROOT / "modules"


@contextmanager
def _module_context(module_dir: Path, extra_sys_paths: tuple[str, ...] = ()):
    added = [str(module_dir)] + [str(module_dir / p) for p in extra_sys_paths]
    # Insert in reverse so the module's own root ends up first.
    for path in reversed(added):
        if path not in sys.path:
            sys.path.insert(0, path)

    real_set_page_config = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None
    try:
        yield
    finally:
        st.set_page_config = real_set_page_config
        for path in added:
            if path in sys.path:
                sys.path.remove(path)


def run_module_app(
    module_dir: Path,
    extra_sys_paths: tuple[str, ...] = (),
    entry: str = "app.py",
) -> None:
    """Execute ``module_dir/entry`` as if ``streamlit run`` had launched it.

    ``run_name="__main__"`` matches what Streamlit itself uses, so any module
    that inspects ``__name__`` behaves identically to running it standalone.
    """
    script = module_dir / entry
    if not script.exists():
        st.error(f"Module entry point not found: `{script.relative_to(REPO_ROOT)}`")
        return

    with _module_context(module_dir, extra_sys_paths):
        runpy.run_path(str(script), run_name="__main__")
