#!/usr/bin/env python3
"""Fail fast when the shared SENPAI pod runtime has drifted."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


EXPECTED = {
    "torch": "2.8.0",
    "vllm": "0.11.0",
}


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def nvidia_smi_check() -> dict[str, Any]:
    path = shutil.which("nvidia-smi")
    if not path:
        return {"status": "fail", "detail": "nvidia-smi is not on PATH"}
    stat = Path(path).stat()
    if stat.st_size == 0:
        return {"status": "fail", "detail": f"nvidia-smi is zero bytes: {path}", "path": path}
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )
    if result.returncode != 0:
        return {"status": "fail", "detail": result.stderr.strip() or "nvidia-smi exited non-zero", "path": path}
    return {"status": "pass", "detail": result.stdout.strip(), "path": path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warn-only", action="store_true")
    args = parser.parse_args()

    checks: list[dict[str, Any]] = [nvidia_smi_check()]
    for package, expected in EXPECTED.items():
        version = package_version(package)
        if version and (version == expected or version.startswith(expected + "+")):
            checks.append({"name": package, "status": "pass", "version": version})
        else:
            checks.append({"name": package, "status": "fail", "version": version, "expected": expected})

    checks.append(
        {
            "name": "pip_guard",
            "status": "pass" if os.environ.get("PIP_REQUIRE_VIRTUALENV") == "true" else "warn",
            "PIP_REQUIRE_VIRTUALENV": os.environ.get("PIP_REQUIRE_VIRTUALENV"),
        }
    )

    report = {"status": "fail" if any(c["status"] == "fail" for c in checks) else "pass", "checks": checks}
    print("SENPAI-RUNTIME-DOCTOR: " + json.dumps(report, sort_keys=True))
    if report["status"] != "pass" and not args.warn_only:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
