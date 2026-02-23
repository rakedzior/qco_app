import os
import sys
import subprocess
import hashlib
import ast
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set

BASE_DIR = Path(__file__).resolve().parent

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
APP_FILE = BASE_DIR / "app_qco.py"
REQ_FILE = BASE_DIR / "requirements.txt"

DEFAULT_VENV_DIR = BASE_DIR / "venv"

STATE_DIR = BASE_DIR / ".launcher_state"

# Your corporate CA bundle
CERT_FILE = BASE_DIR / "certs" / "HSBC_CA_bundle.pem"

STREAMLIT_PORT = "8501"
STREAMLIT_ADDR = "127.0.0.1"

REQ_HASH_FILE = STATE_DIR / ".requirements.sha256"
REQ_GEN_HASH_FILE = BASE_DIR / ".app_imports.sha256"  # tracks app_qco.py hash used for requirements generation


# -------------------------------------------------
# UTIL
# -------------------------------------------------
def run(cmd: List[str], env: Optional[Dict] = None, check: bool = True) -> int:
    print(">", " ".join([str(x) for x in cmd]))
    p = subprocess.run(cmd, cwd=str(BASE_DIR), env=env, check=False)
    if check and p.returncode != 0:
        raise SystemExit(f"[ERROR] Command failed with code {p.returncode}: {' '.join(map(str, cmd))}")
    return p.returncode


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -------------------------------------------------
# REQUIREMENTS GENERATION (AST-based)
# -------------------------------------------------
def _is_stdlib_module(mod: str) -> bool:
    """
    Best-effort stdlib detection.
    On Python 3.10+: sys.stdlib_module_names exists and is reliable.
    """
    root = (mod or "").split(".", 1)[0].strip()
    if not root:
        return True

    # Python 3.10+
    std = getattr(sys, "stdlib_module_names", None)
    if isinstance(std, (set, frozenset)) and root in std:
        return True

    # Fallback list for common stdlib used in your app/launcher
    common_std = {
        "os", "sys", "re", "io", "hashlib", "pathlib", "typing", "datetime", "subprocess",
        "traceback", "json", "time", "math", "csv", "logging", "functools", "itertools",
    }
    return root in common_std


def extract_top_level_imports(py_file: Path) -> Set[str]:
    if not py_file.exists():
        raise SystemExit(f"[ERROR] Cannot generate requirements. File not found: {py_file}")

    src = py_file.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(py_file))

    imports: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = (alias.name or "").split(".", 1)[0]
                if root:
                    imports.add(root)
        elif isinstance(node, ast.ImportFrom):
            # relative imports: skip
            if node.level and node.level > 0:
                continue
            mod = node.module or ""
            root = mod.split(".", 1)[0] if mod else ""
            if root:
                imports.add(root)

    # remove stdlib
    imports = {m for m in imports if not _is_stdlib_module(m)}

    # remove "local" modules (best-effort): if file/dir exists in project
    local_filtered: Set[str] = set()
    for m in imports:
        if (BASE_DIR / f"{m}.py").exists() or (BASE_DIR / m).is_dir():
            continue
        local_filtered.add(m)

    return local_filtered


def map_imports_to_pypi_packages(imports: Set[str]) -> List[str]:
    """
    Map Python import names to PyPI package names (best-effort).
    Add mappings here when import != pip name.
    """
    mapping = {
        # your app
        "streamlit": "streamlit",
        "pandas": "pandas",
        "sqlalchemy": "SQLAlchemy",
        "st_aggrid": "streamlit-aggrid",
        "pyodbc": "pyodbc",
        "openpyxl": "openpyxl",

        # common extras (if you add later)
        "numpy": "numpy",
        "requests": "requests",
    }

    pkgs: Set[str] = set()
    for m in imports:
        pkgs.add(mapping.get(m, m))

    # ensure critical runtime deps are present even if import parser missed something
    pkgs.add("streamlit")
    return sorted(pkgs, key=lambda x: x.lower())


def should_regenerate_requirements() -> bool:
    """
    Regenerate requirements.txt if:
    - requirements.txt does not exist
    - app_qco.py hash changed since last generation
    """
    if not REQ_FILE.exists():
        return True
    if not APP_FILE.exists():
        return False  # will be handled later

    current = sha256_file(APP_FILE)
    prev = REQ_GEN_HASH_FILE.read_text(encoding="utf-8").strip() if REQ_GEN_HASH_FILE.exists() else ""
    return current != prev


def generate_requirements_from_app() -> None:
    if not APP_FILE.exists():
        raise SystemExit(f"[ERROR] app_qco.py not found: {APP_FILE}")

    imports = extract_top_level_imports(APP_FILE)
    pkgs = map_imports_to_pypi_packages(imports)

    content = (
        "# Auto-generated from app_qco.py imports (best-effort)\n"
        "# If a package name differs from the import name, update mapping in launcher.py.\n"
        + "\n".join(pkgs)
        + "\n"
    )

    REQ_FILE.write_text(content, encoding="utf-8")

    # store app hash used for generation
    REQ_GEN_HASH_FILE.write_text(sha256_file(APP_FILE), encoding="utf-8")

    print("[INFO] requirements.txt generated/updated from app_qco.py imports.")
    print("[INFO] Generated packages:")
    for p in pkgs:
        print(f"  - {p}")


# -------------------------------------------------
# VENV + INSTALL
# -------------------------------------------------
def venv_python_candidates(venv_dir: Path) -> List[Path]:
    return [
        venv_dir / "Scripts" / "python.exe",  # Windows
        venv_dir / "Scripts" / "python",      # Windows alternative
        venv_dir / "bin" / "python",          # Linux/macOS
    ]


def find_python_in_venv_dir(venv_dir: Path) -> Optional[Path]:
    for candidate in venv_python_candidates(venv_dir):
        if candidate.exists():
            return candidate
    return None


def ensure_venv_python() -> Path:
    # 1) Reuse currently activated virtual env when available.
    active_venv = os.environ.get("VIRTUAL_ENV", "").strip()
    if active_venv:
        active_py = find_python_in_venv_dir(Path(active_venv))
        if active_py:
            print(f"[INFO] Using active virtual environment: {active_venv}")
            return active_py

    # 2) Reuse local project environments if present.
    for candidate_dir in (BASE_DIR / "venv", BASE_DIR / ".venv"):
        existing = find_python_in_venv_dir(candidate_dir)
        if existing:
            print(f"[INFO] Using existing virtual environment: {candidate_dir}")
            return existing

    # 3) No venv found: create one.
    print(f"[INFO] No virtual environment found. Creating one at: {DEFAULT_VENV_DIR}")
    run([sys.executable, "-m", "venv", str(DEFAULT_VENV_DIR)], check=True)
    created = find_python_in_venv_dir(DEFAULT_VENV_DIR)
    if not created:
        raise SystemExit(f"[ERROR] venv created but python not found in: {DEFAULT_VENV_DIR}")
    return created


def parse_requirements(path: Path) -> List[str]:
    reqs: List[str] = []

    def _parse(p: Path) -> None:
        if not p.exists():
            print(f"[WARN] requirements include not found (skipping): {p}")
            return

        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("-r ") or line.startswith("--requirement "):
                included = line.split(maxsplit=1)[1].strip()
                inc_path = (p.parent / included).resolve()
                _parse(inc_path)
                continue

            if line.startswith("-c ") or line.startswith("--constraint "):
                reqs.append(line)
                continue

            reqs.append(line)

    _parse(path)
    return reqs


def split_pip_global_options(entries: List[str]) -> Tuple[List[str], List[str]]:
    global_prefixes = (
        "--index-url",
        "--extra-index-url",
        "--trusted-host",
        "--find-links",
        "-f",
        "--no-index",
        "--prefer-binary",
        "--only-binary",
        "--no-binary",
        "--use-pep517",
        "--pre",
        "-c",
        "--constraint",
    )

    globals_: List[str] = []
    targets: List[str] = []

    for e in entries:
        if e.startswith(global_prefixes):
            globals_.append(e)
        else:
            targets.append(e)

    return globals_, targets


def pip_install_one(venv_python: Path, target: str, global_opts: List[str]) -> bool:
    cmd = [str(venv_python), "-m", "pip", "install"]
    for opt in global_opts:
        cmd.extend(opt.split())
    cmd.append(target)

    code = run(cmd, check=False)
    return code == 0


def ensure_requirements_installed(venv_python: Path) -> None:
    if not REQ_FILE.exists():
        raise SystemExit(f"[ERROR] requirements.txt not found: {REQ_FILE}")

    current_hash = sha256_file(REQ_FILE)

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    failed_file = STATE_DIR / ".requirements_failed.txt"
    had_prev_failures = failed_file.exists() and failed_file.read_text(encoding="utf-8").strip() != ""

    if REQ_HASH_FILE.exists() and REQ_HASH_FILE.read_text(encoding="utf-8").strip() == current_hash and not had_prev_failures:
        print("[INFO] requirements.txt unchanged — skipping install.")
        return

    print("[INFO] Installing/updating dependencies from requirements.txt...")
    run([str(venv_python), "-m", "pip", "install", "--upgrade", "pip"], check=False)
    run([str(venv_python), "-m", "pip", "install", "--upgrade", "setuptools", "wheel"], check=False)

    entries = parse_requirements(REQ_FILE)
    global_opts, targets = split_pip_global_options(entries)

    failed: List[str] = []
    for t in targets:
        print(f"[INFO] pip install {t}")
        ok = pip_install_one(venv_python, t, global_opts=global_opts)
        if not ok:
            print(f"[WARN] Skipped (failed): {t}")
            failed.append(t)

    REQ_HASH_FILE.write_text(current_hash, encoding="utf-8")

    if failed:
        failed_file.write_text("\n".join(failed) + "\n", encoding="utf-8")
        print("\n[WARN] Some requirements failed and were skipped:")
        for x in failed:
            print(f"  - {x}")
        print("[WARN] They will be retried on the next run.\n")
    else:
        if failed_file.exists():
            failed_file.unlink()
        print("[INFO] Dependencies installed successfully. Stored requirements hash.")

    # Optional: freeze lock for support/debug (not used automatically)
    try:
        lock = run([str(venv_python), "-m", "pip", "freeze"], check=False)
        _ = lock
    except Exception:
        pass


# -------------------------------------------------
# CERT + RUN
# -------------------------------------------------
def build_env_with_cert() -> dict:
    env = os.environ.copy()

    if CERT_FILE.exists():
        env["REQUESTS_CA_BUNDLE"] = str(CERT_FILE)
        env["SSL_CERT_FILE"] = str(CERT_FILE)
        print(f"[INFO] Using certificate: {CERT_FILE}")
    else:
        print(f"[WARN] Certificate not found (continuing): {CERT_FILE}")

    return env


def run_streamlit(venv_python: Path) -> None:
    if not APP_FILE.exists():
        raise SystemExit(f"[ERROR] app_qco.py not found: {APP_FILE}")

    env = build_env_with_cert()

    cmd = [
        str(venv_python),
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.address",
        STREAMLIT_ADDR,
        "--server.port",
        STREAMLIT_PORT,
    ]

    print("[INFO] Starting Streamlit...")
    run(cmd, env=env, check=False)


def main() -> None:
    # 1) Generate requirements (only if needed)
    if should_regenerate_requirements():
        generate_requirements_from_app()
    else:
        print("[INFO] requirements.txt up-to-date (no regeneration needed).")

    # 2) Ensure venv
    venv_python = ensure_venv_python()

    # 3) Install deps
    ensure_requirements_installed(venv_python)

    # 4) Run app
    run_streamlit(venv_python)


if __name__ == "__main__":
    main()
