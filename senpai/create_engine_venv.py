#!/usr/bin/env python3
"""Create an isolated per-PR Python venv for optional serving backends."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import venv
from pathlib import Path


ENGINE_PACKAGES = {
    "sglang": ["sglang[all]"],
    "tgi": [],
    "tensorrt-llm": [],
    "custom": [],
}


def safe_slug(raw: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip("-") or "engine"


def default_path(engine: str, pr: str | None) -> Path:
    suffix = safe_slug(f"{engine}-pr-{pr}" if pr else engine)
    return Path(os.environ.get("INFERENCE_BENCH_ENGINE_VENV_ROOT", "/tmp/inferencebench-engine-venvs")) / suffix


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default="custom", choices=sorted(ENGINE_PACKAGES))
    parser.add_argument("--pr", help="PR number or branch slug for the venv name")
    parser.add_argument("--path", help="Explicit venv path")
    parser.add_argument(
        "--package",
        action="append",
        default=[],
        help="Additional pip package spec. Repeat for multiple packages.",
    )
    parser.add_argument("--extra-index-url", action="append", default=[])
    parser.add_argument("--force", action="store_true", help="Delete any existing venv first")
    args = parser.parse_args()

    path = Path(args.path).expanduser().resolve() if args.path else default_path(args.engine, args.pr).resolve()
    if path.exists() and args.force:
        import shutil

        shutil.rmtree(path)
    if not path.exists():
        venv.EnvBuilder(with_pip=True, clear=False, symlinks=True).create(path)

    python = path / "bin" / "python"
    pip = path / "bin" / "pip"
    env = os.environ.copy()
    env["PIP_REQUIRE_VIRTUALENV"] = "false"
    run([str(python), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], env)
    packages = [*ENGINE_PACKAGES[args.engine], *args.package]
    if packages:
        cmd = [str(pip), "install"]
        for index in args.extra_index_url:
            cmd.extend(["--extra-index-url", index])
        cmd.extend(packages)
        run(cmd, env)

    print(f"[engine-venv] path={path}")
    print(f"[engine-venv] activate: source {path}/bin/activate")
    print(f"[engine-venv] python: {python}")


if __name__ == "__main__":
    main()
