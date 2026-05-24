#!/usr/bin/env python3
"""Shared-pod GPU slot lease for InferenceBench SENPAI runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import socket
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator


DEFAULT_SLOT_FILE = "/tmp/inferencebench-gpu-slot.json"


def now() -> float:
    return time.time()


def utc(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now() if ts is None else ts))


def slot_path(raw: str | None) -> Path:
    return Path(raw or os.environ.get("INFERENCE_BENCH_GPU_SLOT_FILE", DEFAULT_SLOT_FILE)).expanduser().resolve()


@contextmanager
def locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_lease(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def write_lease(path: Path, lease: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(lease, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def lease_is_stale(lease: dict[str, Any] | None) -> bool:
    if not lease:
        return True
    expires_at = lease.get("expires_at")
    return not isinstance(expires_at, (int, float)) or float(expires_at) <= now()


def lease_summary(lease: dict[str, Any] | None) -> str:
    if not lease:
        return "GPU slot is free"
    stale = " stale" if lease_is_stale(lease) else ""
    owner = lease.get("owner", "unknown")
    pr = lease.get("pr") or "-"
    scenario = lease.get("scenario") or "-"
    expires_at = lease.get("expires_at")
    expires = utc(float(expires_at)) if isinstance(expires_at, (int, float)) else "unknown"
    return f"GPU slot held{stale}: owner={owner} pr={pr} scenario={scenario} expires_at={expires}"


def build_lease(args: argparse.Namespace, command: list[str] | None = None) -> dict[str, Any]:
    started = now()
    return {
        "owner": args.owner,
        "pr": args.pr,
        "scenario": args.scenario,
        "purpose": args.purpose,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "command": command or [],
        "created_at": started,
        "created_at_utc": utc(started),
        "updated_at": started,
        "updated_at_utc": utc(started),
        "ttl_s": args.ttl_s,
        "expires_at": started + args.ttl_s,
        "expires_at_utc": utc(started + args.ttl_s),
    }


def acquire(path: Path, args: argparse.Namespace, command: list[str] | None = None) -> dict[str, Any]:
    deadline = None if args.wait else now()
    if args.wait_timeout_s is not None:
        deadline = now() + args.wait_timeout_s

    while True:
        with locked(path):
            lease = read_lease(path)
            if lease_is_stale(lease):
                new_lease = build_lease(args, command)
                write_lease(path, new_lease)
                return new_lease
            if lease and lease.get("owner") == args.owner and lease.get("pr") == args.pr:
                new_lease = build_lease(args, command)
                new_lease["replaced_previous_from_same_owner"] = lease
                write_lease(path, new_lease)
                return new_lease

        if not args.wait or (deadline is not None and now() >= deadline):
            raise SystemExit(lease_summary(lease))
        print(lease_summary(lease), file=sys.stderr)
        time.sleep(args.poll_s)


def release(path: Path, owner: str, *, pr: str | None = None, force: bool = False) -> bool:
    with locked(path):
        lease = read_lease(path)
        if not lease:
            return False
        owner_match = lease.get("owner") == owner
        pr_match = pr is None or lease.get("pr") == pr
        if not force and not (owner_match and pr_match):
            raise SystemExit(f"refusing to release someone else's lease: {lease_summary(lease)}")
        path.unlink(missing_ok=True)
        return True


def heartbeat(path: Path, owner: str, *, pr: str | None, ttl_s: int) -> bool:
    with locked(path):
        lease = read_lease(path)
        if not lease:
            return False
        if lease.get("owner") != owner or (pr is not None and lease.get("pr") != pr):
            return False
        ts = now()
        lease["updated_at"] = ts
        lease["updated_at_utc"] = utc(ts)
        lease["ttl_s"] = ttl_s
        lease["expires_at"] = ts + ttl_s
        lease["expires_at_utc"] = utc(ts + ttl_s)
        write_lease(path, lease)
        return True


def heartbeat_loop(path: Path, owner: str, pr: str | None, ttl_s: int, stop: threading.Event) -> None:
    while not stop.wait(min(max(ttl_s // 4, 15), 120)):
        heartbeat(path, owner, pr=pr, ttl_s=ttl_s)


def cmd_status(args: argparse.Namespace) -> int:
    path = slot_path(args.slot_file)
    lease = read_lease(path)
    if args.json:
        print(json.dumps({"slot_file": str(path), "lease": lease, "stale": lease_is_stale(lease)}, sort_keys=True))
    else:
        print(lease_summary(lease))
        print(f"slot_file={path}")
    return 0


def cmd_acquire(args: argparse.Namespace) -> int:
    path = slot_path(args.slot_file)
    lease = acquire(path, args)
    print(f"acquired GPU slot: {lease_summary(lease)}")
    return 0


def cmd_release(args: argparse.Namespace) -> int:
    path = slot_path(args.slot_file)
    released = release(path, args.owner, pr=args.pr, force=args.force)
    print("released GPU slot" if released else "GPU slot was already free")
    return 0


def cmd_heartbeat(args: argparse.Namespace) -> int:
    path = slot_path(args.slot_file)
    ok = heartbeat(path, args.owner, pr=args.pr, ttl_s=args.ttl_s)
    print("heartbeat updated" if ok else "heartbeat ignored; lease not owned by this process")
    return 0 if ok else 1


def cmd_run(args: argparse.Namespace) -> int:
    path = slot_path(args.slot_file)
    lease = acquire(path, args, command=args.command)
    print(f"acquired GPU slot: {lease_summary(lease)}", flush=True)
    stop = threading.Event()
    beat = threading.Thread(target=heartbeat_loop, args=(path, args.owner, args.pr, args.ttl_s, stop), daemon=True)
    beat.start()
    try:
        return subprocess.run(args.command, check=False).returncode
    finally:
        stop.set()
        release(path, args.owner, pr=args.pr, force=False)
        print("released GPU slot", flush=True)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--slot-file", help=f"Lease JSON path (default: {DEFAULT_SLOT_FILE})")


def add_owner_common(parser: argparse.ArgumentParser) -> None:
    add_common(parser)
    parser.add_argument("--owner", required=True, help="Student/advisor name that owns the slot")
    parser.add_argument("--pr", help="PR number or branch for this lease")
    parser.add_argument("--scenario", help="Scenario A, B, C, or D")
    parser.add_argument("--purpose", default="benchmark", help="Short purpose shown in status")
    parser.add_argument("--ttl-s", type=int, default=7200, help="Lease TTL refreshed by heartbeat/run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command_name", required=True)

    status = sub.add_parser("status", help="Show the current GPU lease")
    add_common(status)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    acquire_parser = sub.add_parser("acquire", help="Acquire the GPU slot")
    add_owner_common(acquire_parser)
    acquire_parser.add_argument("--wait", action="store_true", help="Wait until the slot is free")
    acquire_parser.add_argument("--wait-timeout-s", type=int)
    acquire_parser.add_argument("--poll-s", type=int, default=15)
    acquire_parser.set_defaults(func=cmd_acquire)

    release_parser = sub.add_parser("release", help="Release the GPU slot")
    add_common(release_parser)
    release_parser.add_argument("--owner", required=True)
    release_parser.add_argument("--pr")
    release_parser.add_argument("--force", action="store_true")
    release_parser.set_defaults(func=cmd_release)

    heartbeat_parser = sub.add_parser("heartbeat", help="Refresh a held GPU slot")
    add_common(heartbeat_parser)
    heartbeat_parser.add_argument("--owner", required=True)
    heartbeat_parser.add_argument("--pr")
    heartbeat_parser.add_argument("--ttl-s", type=int, default=7200)
    heartbeat_parser.set_defaults(func=cmd_heartbeat)

    run_parser = sub.add_parser("run", help="Run a command while holding the GPU slot")
    add_owner_common(run_parser)
    run_parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=True)
    run_parser.add_argument("--wait-timeout-s", type=int)
    run_parser.add_argument("--poll-s", type=int, default=15)
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    run_parser.set_defaults(func=cmd_run)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "command_name", "") == "run":
        if args.command and args.command[0] == "--":
            args.command = args.command[1:]
        if not args.command:
            parser.error("run requires a command after --")
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
