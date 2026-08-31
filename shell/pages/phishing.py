"""Social Engineering & Phishing -- runs modules/phishing/app.py unmodified.

No extra sys.path entries are needed: the module's app.py already inserts its
own generator/ and detector/ directories using __file__-relative paths, so it
self-heals wherever it is run from.
"""

from shell.loader import MODULES, run_module_app

run_module_app(MODULES / "phishing")
