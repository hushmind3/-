"""Project paths and web assets."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_PATH = Path(__file__).resolve().parents[1] / "web_dashboard.html"
ASSET_DIR = Path(__file__).with_name("assets")
PAGE = DASHBOARD_PATH.read_text(encoding="utf-8")

ASSET_NAMES = frozenset(("dashboard.css", "core.js", "controls.js", "accounts.js",
                        "learning.js", "portfolio.js", "operations.js", "startup.js"))


def dashboard_asset(route):
    """Read an explicitly published asset; arbitrary paths are never accepted."""
    if not route.startswith("/assets/"):
        return None
    name = route.removeprefix("/assets/")
    if name not in ASSET_NAMES:
        return None
    try:
        body = (ASSET_DIR / name).read_text(encoding="utf-8")
    except OSError:
        return None
    mime = "text/javascript; charset=utf-8" if name.endswith(".js") else "text/css; charset=utf-8"
    return body, mime
