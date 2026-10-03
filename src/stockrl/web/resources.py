"""Project paths and web assets."""
from pathlib import Path
import mimetypes

ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_PATH = Path(__file__).with_name("dist") / "index.html"
ASSET_DIR = DASHBOARD_PATH.parent / "assets"
PAGE = "<!doctype html><html lang='ko'><meta charset='utf-8'><title>UI build 필요</title><p>frontend에서 npm run build를 실행하세요.</p></html>"
ASSET_NAMES = frozenset(path.name for path in ASSET_DIR.glob("*") if path.is_file())

def dashboard_asset(route):
    """Serve only files inside the Vite build's assets directory."""
    if not route.startswith("/assets/"):
        return None
    name = route.removeprefix("/assets/")
    target = (ASSET_DIR / name).resolve()
    if not target.is_relative_to(ASSET_DIR.resolve()) or not target.is_file():
        return None
    try:
        body = target.read_bytes()
    except OSError:
        return None
    mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    if target.suffix == ".js":
        mime = "text/javascript"
    if mime.startswith("text/"):
        mime += "; charset=utf-8"
    return body, mime
