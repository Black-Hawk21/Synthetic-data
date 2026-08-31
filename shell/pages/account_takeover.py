"""Authentication & Account Takeover -- runs modules/account_takeover/app.py.

The module's app.py and all four pipeline scripts do `import config as cfg`,
a bare top-level name, and app.py reaches for `05_mitigate_and_demo` through
importlib (a name starting with a digit can never be a normal import). Both
resolve because the loader puts the module's own directory on sys.path.
"""

from shell.loader import MODULES, run_module_app

run_module_app(MODULES / "account_takeover")
