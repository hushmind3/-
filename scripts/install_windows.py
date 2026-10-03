"""Recreate local Python environments and paths after downloading the project.

No model load, replay copy, account reset or trading is performed here.
"""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "artifacts" / "experts"


def install_environment(name, requirements, *, toto=False):
    env = ASSETS / name
    python = env / "Scripts" / "python.exe"
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    command = [str(python), "-m", "pip", "install", "--disable-pip-version-check",
               "--extra-index-url", "https://download.pytorch.org/whl/cu132",
               "-r", str(ROOT / requirements), "-e", str(ROOT)]
    if toto:
        command += ["transformers==4.51.3", "lightning==2.4.0", "gluonts==0.16.2", "jaxtyping==0.3.11"]
    subprocess.run(command, cwd=ROOT, check=True)


def prepare_local_paths():
    sys.path.insert(0, str(ROOT / "src"))
    from stockrl.paths import DEFAULT_MODEL_DIR, default_runtime_dir
    from stockrl.expert_registry import ensure_registry
    DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    default_runtime_dir().mkdir(parents=True, exist_ok=True)
    target = ROOT / "configs" / "local" / "web_settings.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes((ROOT / "configs" / "web_settings.default.json").read_bytes())
    ensure_registry(ROOT / "runtime" / "trading_moe" / "registry.json")
    print("Local paths prepared. DB schema is created by the replay/paper engine on first use.")
    print("Model files: " + str(DEFAULT_MODEL_DIR))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="prepare paths without installing packages")
    args = parser.parse_args()
    os.environ["PYTHONUTF8"] = "1"
    if not args.prepare_only:
        install_environment("venv", "requirements/moe.txt")
        install_environment("venv-toto", "requirements/moe-toto.txt", toto=True)
    prepare_local_paths()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
