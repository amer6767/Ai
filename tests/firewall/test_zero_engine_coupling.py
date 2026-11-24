"""Engine-Isolation Firewall Tests.

NON-NEGOTIABLE DESIGN PRINCIPLE:
The finished AI must NEVER import browser-automation engines (playwright/selenium/cdp),
read text.html, open game WebSockets, or inspect internal JS engine variables at runtime.
Only Zone A offline research (`research/`) and `display_manager.py` (which merely starts
a local HTTP static file server & plain Chrome window) may reference the static file path.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RUNTIME_PACKAGES = [
    "vision",
    "perception",
    "map_analysis",
    "strategic_brain",
    "reinforcement_learning",
    "self_play",
    "memory",
    "evaluation",
    "models",
    "computer_control",
    "core",
]

FORBIDDEN_IMPORTS = {
    "playwright",
    "selenium",
    "pyppeteer",
    "cdp",
    "chrome_devtools",
    "websocket",
    "research",
}

FORBIDDEN_CALL_SUBSTRINGS = (
    "evaluate_js",
    "execute_script",
    "remote-debugging-port",
    "Runtime.evaluate",
    "window.aD",
    "ag.h3",
)


def _iter_runtime_py_files() -> list[Path]:
    files: list[Path] = []
    for pkg in RUNTIME_PACKAGES:
        pkg_dir = ROOT / pkg
        if pkg_dir.exists():
            files.extend(sorted(pkg_dir.rglob("*.py")))
    return files


def test_no_forbidden_engine_coupling_imports() -> None:
    violations: list[str] = []
    for path in _iter_runtime_py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in FORBIDDEN_IMPORTS:
                        violations.append(f"{path.relative_to(ROOT)}:{node.lineno} imports '{alias.name}'")
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in FORBIDDEN_IMPORTS:
                    violations.append(f"{path.relative_to(ROOT)}:{node.lineno} imports from '{node.module}'")
    assert not violations, "Engine-Isolation Firewall VIOLATION:\n" + "\n".join(violations)


def test_no_js_injection_or_cdp_strings_in_runtime() -> None:
    violations: list[str] = []
    for path in _iter_runtime_py_files():
        text = path.read_text(encoding="utf-8")
        for forbidden in FORBIDDEN_CALL_SUBSTRINGS:
            if forbidden in text:
                violations.append(f"{path.relative_to(ROOT)} contains forbidden token '{forbidden}'")
    assert not violations, "Engine-Isolation Firewall VIOLATION:\n" + "\n".join(violations)


def test_runtime_never_reads_text_html_source() -> None:
    """Only display_manager.py / config.py may mention text.html for launching the external server."""
    allowed = {"display_manager.py", "config.py", "live_browser_env.py"}
    violations: list[str] = []
    for path in _iter_runtime_py_files():
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "text.html" in text:
            violations.append(f"{path.relative_to(ROOT)} references 'text.html'")
    assert not violations, "Runtime AI modules must not read text.html:\n" + "\n".join(violations)
