#!/usr/bin/env python3
"""Audit tracked source files before publishing; ignored local data are never read."""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".md", ".txt", ".cff", ".ini"}
SPECIAL_NAMES = {".gitignore", "LICENSE"}
PRIVATE_PATH = re.compile(r"/(?:SSDHome|home|localhome|mnt|scratch)/[\w/.-]+")
FIELD_HEADER = re.compile(r"\b\d{2,6}-\d+\.\d+\b")
SECRET = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def audit_paths(root: Path, paths):
    errors = []
    for name in paths:
        rel = Path(name)
        path = root / rel
        if rel.parts[0] in {"data", "runs", "results", "private", "artifacts"} or (
            rel.parts[:2] == ("Delphi", "data") and len(rel.parts) > 3
        ):
            errors.append(f"{name}: generated/restricted data directory")
        if path.is_symlink():
            errors.append(f"{name}: symlink")
            continue
        if not path.is_file():
            errors.append(f"{name}: missing tracked file")
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name not in SPECIAL_NAMES:
            errors.append(f"{name}: unapproved file type")
            continue
        if path.stat().st_size > 256 * 1024:
            errors.append(f"{name}: unexpectedly large source file")
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError:
            errors.append(f"{name}: binary/non-UTF-8 content")
            continue
        if "\x00" in content:
            errors.append(f"{name}: binary content")
        for rule, pattern in [("machine-specific absolute path", PRIVATE_PATH),
                              ("cohort-specific field header", FIELD_HEADER), ("possible credential", SECRET)]:
            if pattern.search(content):
                errors.append(f"{name}: {rule}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    raw = subprocess.check_output(["git", "ls-files", "--cached", "-z"], cwd=args.root)
    paths = [name for name in raw.decode().split("\0") if name]
    if not paths:
        raise SystemExit("No tracked files to audit; stage the public source files first")
    errors = audit_paths(args.root, paths)
    if errors:
        raise SystemExit("Public release audit failed:\n" + "\n".join(errors))
    print(f"Public release audit passed: {len(paths)} tracked text files, no data, symlinks, weights or detected secrets")


if __name__ == "__main__":
    main()
