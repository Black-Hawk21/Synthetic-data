"""Chargeback fraud (Aegis) -- runs modules/chargeback/app.py unmodified."""

from shell.loader import MODULES, run_module_app

run_module_app(MODULES / "chargeback")
