"""Offline static analysis decoder for Territorial.io (text.html).

ZONE A RESEARCH TOOL ONLY.
This module must NEVER be imported by any runtime AI package.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_source(html_path: str | Path = "text.html") -> str:
    return Path(html_path).read_text(encoding="utf-8")


def extract_string_table(src: str) -> list[str]:
    match = re.search(r"var S=(\[.*?\]);", src, re.S)
    if not match:
        raise ValueError("Could not locate string table var S=[...] in text.html")
    return json.loads(match.group(1))


def substitute_string_table(src: str, table: list[str]) -> str:
    def _repl(m: re.Match[str]) -> str:
        idx = int(m.group(1))
        if 0 <= idx < len(table):
            return json.dumps(table[idx])
        return m.group(0)

    return re.sub(r"\bS\[(\d+)\]", _repl, src)


def list_constructor_blocks(decoded_src: str) -> list[dict[str, Any]]:
    """Index top-level function definitions and their approximate line spans."""
    lines = decoded_src.splitlines()
    blocks: list[dict[str, Any]] = []
    fn_pat = re.compile(r"function\s+([a-zA-Z0-9_$]+)\s*\(([^)]*)\)\s*\{")
    for idx, line in enumerate(lines, start=1):
        for m in fn_pat.finditer(line):
            blocks.append(
                {
                    "name": m.group(1),
                    "args": m.group(2),
                    "line": idx,
                    "snippet": line[m.start() : m.start() + 140],
                }
            )
    return blocks


def search_keywords(decoded_src: str, keywords: list[str], context: int = 2) -> dict[str, list[dict[str, Any]]]:
    lines = decoded_src.splitlines()
    results: dict[str, list[dict[str, Any]]] = {}
    for kw in keywords:
        pat = re.compile(re.escape(kw), re.IGNORECASE)
        hits: list[dict[str, Any]] = []
        for idx, line in enumerate(lines):
            if pat.search(line):
                start = max(0, idx - context)
                end = min(len(lines), idx + context + 1)
                hits.append(
                    {
                        "line": idx + 1,
                        "excerpt": "\n".join(f"{j+1}: {lines[j][:220]}" for j in range(start, end)),
                    }
                )
        results[kw] = hits
    return results


if __name__ == "__main__":
    raw = load_source()
    table = extract_string_table(raw)
    decoded = substitute_string_table(raw, table)
    blocks = list_constructor_blocks(decoded)
    print(f"String table entries: {len(table)}")
    print(f"Total function definitions indexed: {len(blocks)}")
    kws = [
        "Single",
        "Very Easy",
        "Easy",
        "Normal",
        "Hard",
        "Very Hard",
        "Impossible",
        "Interest",
        "Boat",
        "Attack",
        "Peace",
        "Spawn",
        "Victory",
        "Winner",
        "Crown",
        "Map",
        "Slider",
        "keydown",
        "mousedown",
        "wheel",
    ]
    found = search_keywords(decoded, kws, context=0)
    for k, v in found.items():
        print(f"  {k:12s}: {len(v):3d} hits (first lines: {[h['line'] for h in v[:6]]})")
