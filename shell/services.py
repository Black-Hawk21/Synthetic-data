"""Probe and launch the out-of-process services that modules/identity needs.

modules/identity is the one module that is not a Streamlit app: it is a
FastAPI backend on :8000 plus a React/Vite dev server on :5173. The shell
cannot host it in-process, so this module reports on it and can start it.

The Vite dev server is not optional scaffolding here -- frontend/src/services
/api.js uses relative base URLs ("/api" and "/"), which only resolve behind
the proxy declared in frontend/vite.config.js. Serving the built bundle
without an equivalent proxy gives 404s on every call.
"""

from __future__ import annotations

import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
IDENTITY_DIR = REPO_ROOT / "modules" / "identity"

BACKEND_PORT = 8000
FRONTEND_PORT = 5173


@dataclass(frozen=True)
class Service:
    name: str
    port: int
    command: list[str]
    cwd: Path
    note: str

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}"

    def is_up(self, timeout: float = 0.35) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return sock.connect_ex(("127.0.0.1", self.port)) == 0


# uvicorn must run with cwd=modules/identity: every import inside the module
# is absolute (`from backend.X import ...`), so the module root has to be the
# import root.
BACKEND = Service(
    name="Identity backend (FastAPI)",
    port=BACKEND_PORT,
    command=["uvicorn", "backend.main:app", "--port", str(BACKEND_PORT)],
    cwd=IDENTITY_DIR,
    note="Serves /api/* and /health.",
)

FRONTEND = Service(
    name="Identity frontend (Vite)",
    port=FRONTEND_PORT,
    command=["npm", "run", "dev"],
    cwd=IDENTITY_DIR / "frontend",
    note="Proxies /api and /health to the backend. Needs `npm install` first.",
)

SERVICES = (BACKEND, FRONTEND)


def start(service: Service) -> subprocess.Popen:
    """Launch a service detached enough to outlive one Streamlit rerun."""
    return subprocess.Popen(
        service.command,
        cwd=str(service.cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def frontend_deps_installed() -> bool:
    return (IDENTITY_DIR / "frontend" / "node_modules").is_dir()


def missing_backend_package() -> Path | None:
    """Return the path of the absent ``backend/data`` package, if it is absent.

    ``backend/data/`` holds two hand-written source modules, ``generator.py``
    and ``schemas.py``. It is NOT the module's generated data directory -- that
    is ``modules/identity/data/``, which backend/config.py creates on import.

    The package was lost because the repository's root .gitignore used to
    carry a bare ``data/`` rule, which matches at any depth. The rule is fixed
    now, but the two files were never committed under it and have to be
    restored from the author's working tree. Twelve files import them at
    module scope, so the backend cannot start without them.
    """
    package = IDENTITY_DIR / "backend" / "data"
    if (package / "generator.py").exists() and (package / "schemas.py").exists():
        return None
    return package
