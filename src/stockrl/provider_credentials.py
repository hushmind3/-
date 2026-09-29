"""Kiwoom market-data credentials stored in the operating system vault."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .paths import ensure_project_path

SERVICE = "stockrl.market-data"
PROVIDERS = {"kiwoom": "\ud0a4\uc6c0\uc99d\uad8c"}


def _vault():
    try:
        import keyring
    except ImportError as exc:
        raise RuntimeError("keyring \ud328\ud0a4\uc9c0\uac00 \ud544\uc694\ud569\ub2c8\ub2e4.") from exc
    vault = keyring.get_keyring()
    name = type(vault).__name__.lower()
    if getattr(vault, "priority", 0) <= 0 or "fail" in name or "plaintext" in name:
        raise RuntimeError("\uc6b4\uc601\uccb4\uc81c \ubcf4\uc548 \uc800\uc7a5\uc18c\ub97c \uc0ac\uc6a9\ud560 \uc218 \uc5c6\uc2b5\ub2c8\ub2e4.")
    return keyring


def _settings_path(runtime: Path | None = None) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    local = project_root / "configs" / "local"
    configured = Path(os.environ.get("STOCKRL_LOCAL_CONFIG_DIR", local)).expanduser()
    settings_dir = ensure_project_path(configured, "provider settings")
    return ensure_project_path(settings_dir / "provider_settings.json", "provider settings")


def _migration_credentials(runtime: Path, environment: str) -> dict:
    """Read an explicitly created cross-machine migration file when present."""
    try:
        runtime_dir = ensure_project_path(runtime, "runtime")
        payload = json.loads((runtime_dir / "credentials_migration.json").read_text(encoding="utf-8"))
        selected = payload.get(environment, {}) if isinstance(payload, dict) else {}
        return {field: str(selected.get(field, "") or "")
                for field in ("app_key", "secret", "account")}
    except (OSError, json.JSONDecodeError, AttributeError):
        return {field: "" for field in ("app_key", "secret", "account")}


def _write_settings(runtime: Path, settings: dict) -> None:
    path = _settings_path(runtime)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def read_settings(runtime: Path) -> dict:
    try:
        return json.loads(_settings_path(runtime).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"provider": "yahoo", "environment": "paper"}


def get_credentials(runtime: Path, environment: str | None = None) -> dict:
    """Read only the selected environment; retain legacy vault entries."""
    cfg = read_settings(runtime)
    environment = environment or cfg.get("environment", "paper")
    migrated = _migration_credentials(runtime, environment)
    if migrated["app_key"] and migrated["secret"]:
        return migrated
    vault = _vault()
    fields = ("app_key", "secret", "account")
    scoped = {field: vault.get_password(SERVICE, f"kiwoom:{environment}:{field}") or ""
              for field in fields}
    if (scoped["app_key"] or scoped["secret"]) or cfg.get("environment") != environment:
        return scoped
    return {field: vault.get_password(SERVICE, f"kiwoom:{field}") or ""
            for field in fields}


def public_status(runtime: Path) -> dict:
    cfg = read_settings(runtime)
    provider = cfg.get("provider", "yahoo")
    state = {
        "provider": provider,
        "provider_name": PROVIDERS.get(provider, "\uacf5\uac1c \uc2dc\uc138 (Yahoo)"),
        "environment": cfg.get("environment", "paper"),
        "saved": False,
        "has_app_key": False,
        "has_secret": False,
        "has_account": False,
        "last_test": cfg.get("last_test"),
    }
    if provider == "kiwoom":
        try:
            saved = get_credentials(runtime, state["environment"])
            for key, field in (("app_key", "has_app_key"), ("secret", "has_secret"),
                               ("account", "has_account")):
                state[field] = bool(saved[key])
            state["saved"] = state["has_app_key"] and state["has_secret"]
        except RuntimeError as exc:
            state["vault_error"] = str(exc)
    return state


def save_credentials(runtime: Path, provider: str, environment: str, app_key: str,
                     secret: str, account: str = "") -> dict:
    if provider != "kiwoom":
        raise ValueError("\ud0a4\uc6c0\uc99d\uad8c\ub9cc \uc9c0\uc6d0\ud569\ub2c8\ub2e4.")
    if environment not in ("paper", "real"):
        raise ValueError("\ubc1c\uae09 \ud658\uacbd\uc740 \ubaa8\uc758\ud22c\uc790 \ub610\ub294 \uc2e4\uc804\uc774\uc5b4\uc57c \ud569\ub2c8\ub2e4.")
    vault = _vault()
    old = read_settings(runtime)
    new_pair = bool(app_key.strip() or secret.strip())
    previous = get_credentials(runtime, environment)
    app_key = app_key.strip() or previous["app_key"]
    secret = secret.strip() or previous["secret"]
    account = account.strip() if new_pair else (account.strip() or previous["account"])
    if not app_key or not secret:
        raise ValueError("App Key\uc640 Secret Key\ub97c \ubaa8\ub450 \uc785\ub825\ud574 \uc8fc\uc138\uc694.")
    vault.set_password(SERVICE, f"kiwoom:{environment}:app_key", app_key)
    vault.set_password(SERVICE, f"kiwoom:{environment}:secret", secret)
    if account:
        vault.set_password(SERVICE, f"kiwoom:{environment}:account", account)
    else:
        try:
            vault.delete_password(SERVICE, f"kiwoom:{environment}:account")
        except Exception:
            pass
    settings = {
        "provider": "kiwoom",
        "environment": environment,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "last_test": old.get("last_test") if not new_pair and
                     old.get("provider") == provider and old.get("environment") == environment else None,
    }
    _write_settings(runtime, settings)
    return public_status(runtime)


def test_connection(runtime: Path, provider_override: str | None = None,
                    environment_override: str | None = None,
                    app_key_override: str | None = None,
                    secret_override: str | None = None,
                    persist: bool = True) -> dict:
    import requests

    settings = read_settings(runtime)
    provider = provider_override or settings.get("provider", "yahoo")
    environment = environment_override or settings.get("environment", "paper")
    if provider != "kiwoom":
        raise ValueError("\ud0a4\uc6c0\uc99d\uad8c\uc744 \uc120\ud0dd\ud574 \uc8fc\uc138\uc694.")
    if environment not in ("paper", "real"):
        raise ValueError("\ubc1c\uae09 \ud658\uacbd\uc744 \uc120\ud0dd\ud574 \uc8fc\uc138\uc694.")
    if (app_key_override is None) != (secret_override is None):
        raise ValueError("App Key and Secret Key must be supplied together")
    if app_key_override is None:
        saved = get_credentials(runtime, environment)
        app_key = saved["app_key"]
        secret = saved["secret"]
    else:
        app_key = app_key_override.strip()
        secret = secret_override.strip()
    if not app_key or not secret:
        raise ValueError("\ud0a4\ub97c \uba3c\uc800 \uc800\uc7a5\ud574 \uc8fc\uc138\uc694.")
    host = "https://mockapi.kiwoom.com" if environment == "paper" else "https://api.kiwoom.com"
    response = requests.post(
        host + "/oauth2/token",
        json={"grant_type": "client_credentials", "appkey": app_key, "secretkey": secret},
        headers={"Content-Type": "application/json;charset=UTF-8"},
        timeout=15,
    )
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    ok = response.ok and bool(payload.get("token"))
    code = payload.get("return_code")
    detail = str(payload.get("return_msg") or payload.get("msg") or "")
    for credential in (app_key, secret):
        detail = detail.replace(credential, "[redacted]")
    if ok:
        message = "\uc778\uc99d \uc131\uacf5 \u00b7 \uc811\uc18d \ud1a0\ud070\uc740 \uc800\uc7a5\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4."
    else:
        pieces = ["\uc778\uc99d \uc2e4\ud328"]
        if code is not None:
            pieces.append(f"\ucf54\ub4dc {code}")
        pieces.append(detail or f"HTTP {response.status_code} \u00b7 \uc751\ub2f5 \ud655\uc778 \ud544\uc694")
        message = " \u00b7 ".join(pieces)
    result = {"ok": ok, "time": datetime.now(timezone.utc).isoformat(), "message": message}
    if persist:
        settings["last_test"] = result
        if ok:
            settings.update({"provider": "kiwoom", "environment": environment})
        _write_settings(runtime, settings)
    return result


def connect_credentials(runtime: Path, environment: str, app_key: str,
                        secret: str, account: str = "") -> dict:
    """Validate a new pair before replacing the saved working credentials."""
    result = test_connection(runtime, "kiwoom", environment, app_key, secret, persist=False)
    if result["ok"]:
        save_credentials(runtime, "kiwoom", environment, app_key, secret, account)
        settings = read_settings(runtime)
        settings["last_test"] = result
        _write_settings(runtime, settings)
    return result


def clear_credentials(runtime: Path, provider: str) -> dict:
    if provider != "kiwoom":
        raise ValueError("\ud0a4\uc6c0\uc99d\uad8c \ud0a4\ub9cc \uc0ad\uc81c\ud569\ub2c8\ub2e4.")
    vault = _vault()
    for environment in ("real", "paper", None):
        for field in ("app_key", "secret", "account"):
            key = f"kiwoom:{environment}:{field}" if environment else f"kiwoom:{field}"
            try:
                vault.delete_password(SERVICE, key)
            except Exception:
                pass
    settings = read_settings(runtime)
    if settings.get("provider") == "kiwoom":
        settings.update({"provider": "yahoo", "environment": "paper", "last_test": None})
        _write_settings(runtime, settings)
    return public_status(runtime)
