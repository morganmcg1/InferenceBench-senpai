#!/usr/bin/env python3
"""Shared-pod GPU slot lease for InferenceBench SENPAI runs."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


DEFAULT_SLOT_FILE = "/tmp/inferencebench-gpu-slot.json"
LOST_LEASE_EXIT = 75
RUNTIME_LIMIT_EXIT = 124

MODE_DEFAULTS = {
    "quick": {"ttl_s": 900, "min_remaining_s": 900, "max_runtime_s": 900},
    "full": {"ttl_s": 3600, "min_remaining_s": 1800, "max_runtime_s": 3600},
}


def now() -> float:
    return time.time()


def utc(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now() if ts is None else ts))


def parse_deadline(raw: str | None) -> float | None:
    if not raw:
        return None
    value = raw.strip().strip("'\"")
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc).timestamp()


def deadline_epoch(args: argparse.Namespace) -> float | None:
    explicit = parse_deadline(getattr(args, "deadline_utc", None))
    if explicit is not None:
        return explicit
    explicit = parse_deadline(getattr(args, "deadline_epoch", None))
    if explicit is not None:
        return explicit
    for key in (
        "INFERENCE_BENCH_RUN_DEADLINE_UTC",
        "INFERENCE_BENCH_RUN_DEADLINE_EPOCH",
        "SENPAI_RUN_DEADLINE_UTC",
        "SENPAI_RUN_DEADLINE_EPOCH",
        "KILL_AT_UTC",
        "KILL_AT_EPOCH",
    ):
        parsed = parse_deadline(os.environ.get(key))
        if parsed is not None:
            return parsed
    return None


def apply_slot_policy(args: argparse.Namespace) -> None:
    mode = str(getattr(args, "mode", None) or "custom").lower()
    if mode not in {"custom", "quick", "full"}:
        raise SystemExit(f"unknown GPU slot mode: {mode}")
    args.mode = mode

    ttl_s = int(getattr(args, "ttl_s", None) or 1800)
    min_remaining_s = int(getattr(args, "min_remaining_s", None) or 0)
    max_runtime_s = getattr(args, "max_runtime_s", None)
    max_runtime = int(max_runtime_s) if max_runtime_s is not None else None

    defaults = MODE_DEFAULTS.get(mode)
    if defaults:
        ttl_s = min(ttl_s, defaults["ttl_s"])
        min_remaining_s = max(min_remaining_s, defaults["min_remaining_s"])
        max_runtime = defaults["max_runtime_s"] if max_runtime is None else min(max_runtime, defaults["max_runtime_s"])

    args.ttl_s = ttl_s
    args.min_remaining_s = min_remaining_s
    args.max_runtime_s = max_runtime


def deadline_issue(args: argparse.Namespace) -> str | None:
    deadline = deadline_epoch(args)
    if deadline is None:
        return None
    min_remaining = int(getattr(args, "min_remaining_s", 0) or 0)
    remaining = deadline - now()
    if remaining < min_remaining:
        return (
            f"run deadline {utc(deadline)} leaves {int(max(remaining, 0))}s, "
            f"less than required min_remaining_s={min_remaining}"
        )
    return None


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


def active_gpu_processes() -> list[dict[str, Any]]:
    """Return active NVIDIA compute processes, or [] when nvidia-smi is absent."""

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        return []
    if result.returncode != 0:
        return []

    processes: list[dict[str, Any]] = []
    for raw in result.stdout.splitlines():
        line = raw.strip()
        if not line:
            continue
        fields = [part.strip() for part in line.split(",", 2)]
        if len(fields) < 3:
            continue
        pid_raw, name, memory_raw = fields
        try:
            pid: int | str = int(pid_raw)
        except ValueError:
            pid = pid_raw
        try:
            memory_mb: int | str = int(memory_raw)
        except ValueError:
            memory_mb = memory_raw
        processes.append({"pid": pid, "process_name": name, "used_memory_mb": memory_mb})
    return processes


def gpu_process_summary(processes: list[dict[str, Any]]) -> str:
    return ", ".join(
        f"pid={proc.get('pid')} {proc.get('process_name')} {proc.get('used_memory_mb')}MiB" for proc in processes
    )


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
    if lease.get("owner") == "unleased-gpu-process":
        return "GPU slot has no lease but GPU is busy: " + gpu_process_summary(lease.get("active_gpu_processes") or [])
    stale = " stale" if lease_is_stale(lease) else ""
    owner = lease.get("owner", "unknown")
    pr = lease.get("pr") or "-"
    scenario = lease.get("scenario") or "-"
    expires_at = lease.get("expires_at")
    expires = utc(float(expires_at)) if isinstance(expires_at, (int, float)) else "unknown"
    return f"GPU slot held{stale}: owner={owner} pr={pr} scenario={scenario} expires_at={expires}"


def build_lease(args: argparse.Namespace, command: list[str] | None = None) -> dict[str, Any]:
    started = now()
    deadline = deadline_epoch(args)
    return {
        "lease_id": uuid.uuid4().hex,
        "owner": args.owner,
        "pr": args.pr,
        "scenario": args.scenario,
        "purpose": args.purpose,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "command": command or [],
        "mode": getattr(args, "mode", "custom"),
        "created_at": started,
        "created_at_utc": utc(started),
        "updated_at": started,
        "updated_at_utc": utc(started),
        "ttl_s": args.ttl_s,
        "expires_at": started + args.ttl_s,
        "expires_at_utc": utc(started + args.ttl_s),
        "run_deadline_utc": utc(deadline) if deadline is not None else None,
        "min_remaining_s": int(getattr(args, "min_remaining_s", 0) or 0),
        "max_runtime_s": getattr(args, "max_runtime_s", None),
    }


def acquire(path: Path, args: argparse.Namespace, command: list[str] | None = None) -> dict[str, Any]:
    apply_slot_policy(args)
    issue = deadline_issue(args)
    if issue:
        raise SystemExit(issue)

    wait_deadline = None if args.wait else now()
    if args.wait_timeout_s is not None:
        wait_deadline = now() + args.wait_timeout_s
    run_deadline = deadline_epoch(args)
    min_remaining = int(getattr(args, "min_remaining_s", 0) or 0)
    if run_deadline is not None and min_remaining > 0:
        latest_start = run_deadline - min_remaining
        wait_deadline = latest_start if wait_deadline is None else min(wait_deadline, latest_start)

    while True:
        issue = deadline_issue(args)
        if issue:
            raise SystemExit(issue)
        with locked(path):
            lease = read_lease(path)
            if lease_is_stale(lease):
                active = [] if getattr(args, "ignore_active_gpu", False) else active_gpu_processes()
                if not active:
                    new_lease = build_lease(args, command)
                    write_lease(path, new_lease)
                    return new_lease
                lease = {
                    "owner": "unleased-gpu-process",
                    "pr": "-",
                    "scenario": "-",
                    "purpose": "active GPU process without gpu_slot lease",
                    "active_gpu_processes": active,
                    "expires_at": now() + max(int(getattr(args, "poll_s", 15)), 1),
                }
            elif (
                lease
                and lease.get("owner") == args.owner
                and lease.get("pr") == args.pr
                and getattr(args, "replace_existing", False)
            ):
                new_lease = build_lease(args, command)
                new_lease["replaced_previous_from_same_owner"] = lease
                write_lease(path, new_lease)
                return new_lease

        if not args.wait or (wait_deadline is not None and now() >= wait_deadline):
            raise SystemExit(lease_summary(lease))
        print(lease_summary(lease), file=sys.stderr)
        time.sleep(args.poll_s)


def release(path: Path, owner: str, *, pr: str | None = None, force: bool = False, lease_id: str | None = None) -> bool:
    with locked(path):
        lease = read_lease(path)
        if not lease:
            return False
        owner_match = lease.get("owner") == owner
        pr_match = pr is None or lease.get("pr") == pr
        lease_match = lease_id is None or lease.get("lease_id") == lease_id
        if lease_id is not None and not lease_match:
            return False
        if not force and not (owner_match and pr_match):
            raise SystemExit(f"refusing to release someone else's lease: {lease_summary(lease)}")
        path.unlink(missing_ok=True)
        return True


def heartbeat(path: Path, owner: str, *, pr: str | None, ttl_s: int, lease_id: str | None = None) -> bool:
    with locked(path):
        lease = read_lease(path)
        if not lease:
            return False
        if lease.get("owner") != owner or (pr is not None and lease.get("pr") != pr):
            return False
        if lease_id is not None and lease.get("lease_id") != lease_id:
            return False
        ts = now()
        lease["updated_at"] = ts
        lease["updated_at_utc"] = utc(ts)
        lease["ttl_s"] = ttl_s
        lease["expires_at"] = ts + ttl_s
        lease["expires_at_utc"] = utc(ts + ttl_s)
        write_lease(path, lease)
        return True


def heartbeat_loop(
    path: Path,
    owner: str,
    pr: str | None,
    ttl_s: int,
    lease_id: str | None,
    stop: threading.Event,
    lost: threading.Event,
) -> None:
    while not stop.wait(min(max(ttl_s // 4, 15), 120)):
        if not heartbeat(path, owner, pr=pr, ttl_s=ttl_s, lease_id=lease_id):
            lost.set()
            stop.set()
            return


def terminate_process_group(pgid: int, proc: subprocess.Popen[Any] | None = None, *, grace_s: float = 5.0) -> None:
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return

    deadline = time.monotonic() + grace_s
    while proc is not None and proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except PermissionError:
        return


def cmd_status(args: argparse.Namespace) -> int:
    path = slot_path(args.slot_file)
    lease = read_lease(path)
    gpu_processes = active_gpu_processes()
    if args.json:
        print(
            json.dumps(
                {
                    "slot_file": str(path),
                    "lease": lease,
                    "stale": lease_is_stale(lease),
                    "active_gpu_processes": gpu_processes,
                },
                sort_keys=True,
            )
        )
    else:
        print(lease_summary(lease))
        if gpu_processes:
            print(f"active_gpu_processes={gpu_process_summary(gpu_processes)}")
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
    lease_id = lease.get("lease_id")
    print(f"acquired GPU slot: {lease_summary(lease)}", flush=True)
    stop = threading.Event()
    lost = threading.Event()
    beat = threading.Thread(
        target=heartbeat_loop,
        args=(path, args.owner, args.pr, args.ttl_s, lease_id, stop, lost),
        daemon=True,
    )
    beat.start()
    child_env = os.environ.copy()
    if isinstance(lease_id, str):
        child_env["INFERENCE_BENCH_GPU_SLOT_LEASE_ID"] = lease_id
    child_env["INFERENCE_BENCH_GPU_SLOT_OWNER"] = args.owner
    if args.pr:
        child_env["INFERENCE_BENCH_GPU_SLOT_PR"] = args.pr
    if args.scenario:
        child_env["INFERENCE_BENCH_GPU_SLOT_SCENARIO"] = args.scenario

    proc: subprocess.Popen[Any] | None = None
    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: Any) -> None:
        if proc is not None:
            terminate_process_group(proc.pid, proc, grace_s=5)
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    runtime_deadline = None
    if args.max_runtime_s is not None:
        runtime_deadline = now() + int(args.max_runtime_s)

    try:
        proc = subprocess.Popen(args.command, env=child_env, start_new_session=True)
        while proc.poll() is None:
            if lost.is_set():
                print("lost GPU slot lease; terminating command process group", file=sys.stderr, flush=True)
                terminate_process_group(proc.pid, proc, grace_s=5)
                return LOST_LEASE_EXIT
            if runtime_deadline is not None and now() >= runtime_deadline:
                print(
                    f"GPU slot {args.mode} runtime limit reached after {args.max_runtime_s}s; "
                    "terminating command process group",
                    file=sys.stderr,
                    flush=True,
                )
                terminate_process_group(proc.pid, proc, grace_s=5)
                return RUNTIME_LIMIT_EXIT
            time.sleep(1)
        return int(proc.returncode)
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        stop.set()
        beat.join(timeout=2)
        if proc is not None:
            terminate_process_group(proc.pid, proc, grace_s=2)
        released = release(
            path,
            args.owner,
            pr=args.pr,
            force=False,
            lease_id=lease_id if isinstance(lease_id, str) else None,
        )
        print("released GPU slot" if released else "GPU slot lease already changed; not released", flush=True)


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--slot-file", help=f"Lease JSON path (default: {DEFAULT_SLOT_FILE})")


def add_owner_common(parser: argparse.ArgumentParser) -> None:
    add_common(parser)
    parser.add_argument("--owner", required=True, help="Student/advisor name that owns the slot")
    parser.add_argument("--pr", help="PR number or branch for this lease")
    parser.add_argument("--scenario", help="Scenario A, B, C, or D")
    parser.add_argument("--purpose", default="benchmark", help="Short purpose shown in status")
    parser.add_argument(
        "--mode",
        choices=("custom", "quick", "full"),
        default=os.environ.get("INFERENCE_BENCH_GPU_SLOT_MODE", "custom"),
        help="Lease policy: quick caps runtime/TTL at 15m, full at 60m, custom uses explicit values",
    )
    parser.add_argument("--ttl-s", type=int, default=None, help="Lease TTL refreshed by heartbeat/run")
    parser.add_argument(
        "--max-runtime-s",
        type=int,
        help="Hard wall-clock limit for run commands; quick/full modes set safe defaults",
    )
    parser.add_argument(
        "--deadline-utc",
        help="Optional UTC cutoff time; refuses new work when too little time remains",
    )
    parser.add_argument(
        "--deadline-epoch",
        help="Optional epoch-second cutoff time; overrides env-derived cutoff only when --deadline-utc is absent",
    )
    parser.add_argument(
        "--min-remaining-s",
        type=int,
        default=0,
        help="Minimum wall-clock seconds that must remain before acquiring the slot",
    )
    parser.add_argument(
        "--ignore-active-gpu",
        action="store_true",
        help="Acquire even when nvidia-smi reports unleased compute processes",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Replace an active lease from the same owner and PR",
    )


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
    heartbeat_parser.add_argument("--ttl-s", type=int, default=1800)
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
