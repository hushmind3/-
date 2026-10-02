"""HTTP routes and web-server entrypoint."""
from __future__ import annotations
import json
import gzip
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from ..paths import default_runtime_dir
from .resources import DASHBOARD_PATH, PAGE, ROOT, dashboard_asset
from .runtime import Supervisor
from .workers import handoff

def serve(host: str = "127.0.0.1", port: int = 8766, runtime: str | None = None,
          device: str = "auto", candidate_every: int = 16, fee: float = .001,
          auto_start: bool = True, open_browser: bool = True, horizon: str = "1m",
          config: str = "configs/live_symbols.json", initial_champion: str | None = None,
          model_dir: str | None = None, settings_dir: str | None = None):
    runtime_path = Path(runtime) if runtime is not None else default_runtime_dir()
    if not runtime_path.is_absolute():
        runtime_path = ROOT / runtime_path
    resume_workers=(runtime_path/"web_workers.json").exists()
    supervisor = Supervisor(runtime_path, device, candidate_every, fee, horizon, config, initial_champion,
                            model_dir, settings_dir)
    restart_server_requested = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        server_version = "StockRLWeb/1.0"

        def log_message(self, fmt, *args):
            # Suppress routine HTTP polling access logs; operational events are logged by Supervisor.
            return

        def _send(self, payload, code=200, content_type="application/json; charset=utf-8"):
            data = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False,separators=(",",":")).encode("utf-8")
            compressed="gzip" in self.headers.get("Accept-Encoding","") and len(data)>1024
            if compressed: data=gzip.compress(data,compresslevel=1)
            self.send_response(code); self.send_header("Content-Type", content_type)
            if compressed:
                self.send_header("Content-Encoding","gzip")
                self.send_header("Vary","Accept-Encoding")
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == "/":
                try:
                    page=DASHBOARD_PATH.read_text(encoding="utf-8")
                except OSError:
                    page=PAGE
                return self._send(page, content_type="text/html; charset=utf-8")
            if route.startswith("/assets/"):
                asset = dashboard_asset(route)
                if asset is None:
                    return self._send({"error": "asset not found"}, 404)
                return self._send(asset[0], content_type=asset[1])
            if route == "/api/health":
                return self._send({"ok": True, "service": "stockrl", "port": port,
                    "web_pid":os.getpid(),"worker_pids":{name:p.pid for name,p in supervisor.children.items() if p.poll() is None}})
            if route == "/api/runtime":
                state=(supervisor.profile or supervisor.runtime/supervisor.mode)/"agent"
                from .health import _json
                metrics=_json(state/"metrics.json")
                return self._send({"modes":_json(state/"autonomy.json"),
                    "runtime_updates":metrics.get("runtime_updates"),
                    "gpu_scheduler":metrics.get("gpu_scheduler"),
                    "learning_active_role":metrics.get("learning_active_role"),
                    "champion_training":metrics.get("champion_training"),
                    "candidate_training":metrics.get("candidate_training")})
            if route == "/api/status":
                return self._send(supervisor.status())
            if route in ("/api/experts", "/api/experts/output"):
                from ..expert_registry import read_registry, read_raw_output
                registry = ROOT / "runtime/trading_moe/registry.json"
                try:
                    if route == "/api/experts":
                        return self._send(read_registry(registry))
                    expert_id = parse_qs(urlparse(self.path).query).get("id", [""])[0]
                    return self._send(read_raw_output(registry, expert_id))
                except (OSError, ValueError, KeyError, StopIteration) as exc:
                    return self._send({"error":f"Expert data unavailable: {type(exc).__name__}"}, 404)
            if route == "/api/provider":
                from ..provider_credentials import public_status
                return self._send(public_status(supervisor.runtime))
            if route == "/api/provider/public-ip":
                try:
                    import ipaddress
                    import requests
                    value = requests.get("https://checkip.amazonaws.com/", timeout=8).text.strip()
                    ipaddress.ip_address(value)
                    return self._send({"ip": value})
                except Exception as exc:
                    return self._send({"error": f"Public IP lookup failed: {type(exc).__name__}"}, 502)
            self._send({"error": "not found"}, 404)

        def do_POST(self):
            try:
                n = min(int(self.headers.get("Content-Length", "0")), 4096)
                payload = json.loads(self.rfile.read(n) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._send({"error": "invalid json"}, 400)
            route = urlparse(self.path).path
            if route == "/api/server/restart":
                if restart_server_requested.is_set():
                    return self._send({"error": "Server restart is already in progress."}, 409)
                self._send({"ok": True, "message": "Web server restart accepted; feed and models keep running."})
                threading.Timer(0.5, restart_server_requested.set).start()
                return
            if route == "/api/start":
                result = supervisor.start(payload.get("mode", "live"),payload.get("horizon"))
                return self._send(result, 200 if result.get("ok") else 400)
            if route == "/api/restart":
                result = supervisor.restart(payload.get("mode", "live"),payload.get("horizon"))
                return self._send(result, 200 if result.get("ok") else 400)
            if route == "/api/agent/reload":
                result = supervisor.reload_agent()
                return self._send(result, 200 if result.get("ok") else 409)
            if route == "/api/stop":
                return self._send(supervisor.stop())
            if route == "/api/paper-accounts/reset":
                result = supervisor.reset_paper_accounts()
                return self._send(result, 200 if result.get("ok") else 409)
            if route == "/api/autonomy":
                if not isinstance(payload.get("enabled"),bool):
                    return self._send({"error":"enabled must be a boolean"},400)
                return self._send(supervisor.set_autonomy(payload["enabled"]))
            if route == "/api/modes":
                if any(key in payload and not isinstance(payload[key], bool)
                       for key in ("paper_enabled", "observe_enabled", "learning_enabled")):
                    return self._send({"error":"mode flags must be boolean"},400)
                return self._send(supervisor.set_modes(payload.get("paper_enabled"),
                                                       payload.get("observe_enabled"),
                                                       payload.get("learning_enabled")))
            if route == "/api/feed/reconnect":
                if supervisor.mode != "live" or not supervisor.run_requested:
                    return self._send({"error": "Live market feed is not running."}, 400)
                supervisor.reload_feed()
                return self._send({"ok": True, "message": "Market feed reconnect requested."})
            if route == "/api/provider/save":
                try:
                    from ..provider_credentials import save_credentials
                    result=save_credentials(supervisor.runtime,str(payload.get("provider","")),
                        str(payload.get("environment","paper")),str(payload.get("app_key","")),
                        str(payload.get("secret","")),str(payload.get("account","")))
                    supervisor.reload_feed()
                    return self._send({"ok":True,"provider":result})
                except (ValueError,RuntimeError) as exc:
                    return self._send({"error":str(exc)},400)
            if route == "/api/provider/test":
                try:
                    from ..provider_credentials import test_connection
                    result=test_connection(supervisor.runtime,str(payload.get("provider", "")) or None,str(payload.get("environment", "")) or None)
                    if result.get("ok"):
                        supervisor.reload_feed()
                    return self._send(result)
                except (ValueError,RuntimeError) as exc:
                    return self._send({"error":str(exc)},400)
                except Exception as exc:
                    return self._send({"error":f"Connection check failed: {type(exc).__name__}: {exc}"},502)
            if route == "/api/provider/connect":
                try:
                    from ..provider_credentials import connect_credentials
                    result = connect_credentials(supervisor.runtime,
                        str(payload.get("environment", "real")),
                        str(payload.get("app_key", "")),
                        str(payload.get("secret", "")),
                        str(payload.get("account", "")))
                    if result.get("ok"):
                        supervisor.reload_feed()
                    return self._send(result)
                except (ValueError, RuntimeError) as exc:
                    return self._send({"error": str(exc)}, 400)
                except Exception as exc:
                    return self._send({"error": f"Connection check failed: {type(exc).__name__}: {exc}"}, 502)
            if route == "/api/provider/clear":
                try:
                    from ..provider_credentials import clear_credentials
                    result=clear_credentials(supervisor.runtime,str(payload.get("provider","")))
                    supervisor.reload_feed()
                    return self._send({"ok":True,"provider":result})
                except (ValueError,RuntimeError) as exc:
                    return self._send({"error":str(exc)},400)
            self._send({"error": "not found"}, 404)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    address = server.server_address
    url = f"http://127.0.0.1:{address[1]}/" if host in ("0.0.0.0", "") else f"http://{host}:{address[1]}/"
    if auto_start and not resume_workers:
        supervisor.start(supervisor.mode)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name="stockrl-web")
    thread.start()
    print(f"StockRL web dashboard: {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url, new=1, autoraise=True)
    try:
        while thread.is_alive() and not restart_server_requested.is_set():
            restart_server_requested.wait(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        if restart_server_requested.is_set():
            handoff(supervisor)
        else:
            supervisor.stop()
        server.shutdown(); server.server_close()
        if restart_server_requested.is_set():
            # Only replace the HTTP/controller process. Reattach existing
            # worker identities after exec instead of loading model weights.
            original_argv = getattr(sys, "orig_argv", None)
            if original_argv and len(original_argv) > 1:
                os.execv(sys.executable, [sys.executable, *original_argv[1:]])
            os.execv(sys.executable, [sys.executable, "-m", "stockrl.launch_web", "--server-only"])
