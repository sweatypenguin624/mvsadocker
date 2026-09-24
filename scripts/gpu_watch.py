#!/usr/bin/env python
"""Email me when the GPU frees up.

Polls nvidia-smi and sends one email as soon as a GPU has been free for
long enough to be worth grabbing. Written because GPU 0 on this box sits
occupied for hours at a time and there is no point checking it by hand.

Credentials come from the environment (see --help for setup):
    GPU_WATCH_SMTP_USER   the Gmail address sending the mail
    GPU_WATCH_SMTP_PASS   a Gmail *App Password*, not the account password
    GPU_WATCH_TO          recipient (defaults to GPU_WATCH_SMTP_USER)

Typical use -- start it and forget it:
    nohup env/bin/python scripts/gpu_watch.py > logs/gpu_watch.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import smtplib
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from typing import List, Optional

DEFAULT_TO = "abhaydev2832@gmail.com"
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


@dataclass
class GpuState:
    index: int
    name: str
    used_mib: int
    total_mib: int
    util_pct: int

    @property
    def free_mib(self) -> int:
        return self.total_mib - self.used_mib

    def __str__(self) -> str:
        return (
            f"GPU {self.index} ({self.name}): {self.free_mib:,} MiB free "
            f"of {self.total_mib:,} MiB, {self.util_pct}% utilisation"
        )


def query_gpus() -> List[GpuState]:
    """Read current GPU state. Raises RuntimeError if nvidia-smi fails."""
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout
    except FileNotFoundError:
        raise RuntimeError("nvidia-smi not found -- is this a GPU machine?")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"nvidia-smi failed: {exc.stderr.strip()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("nvidia-smi timed out")

    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        gpus.append(
            GpuState(
                index=int(parts[0]),
                name=parts[1],
                used_mib=int(parts[2]),
                total_mib=int(parts[3]),
                util_pct=int(parts[4]),
            )
        )
    if not gpus:
        raise RuntimeError(f"Could not parse any GPU from nvidia-smi output: {out!r}")
    return gpus


def running_processes() -> str:
    """Who is currently holding the GPU -- included in the email so it's
    clear whether the job that freed up was yours."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return out or "(no compute processes)"
    except Exception:
        return "(could not read process list)"


def is_free(gpu: GpuState, min_free_mib: int, max_util_pct: int) -> bool:
    """Free = enough spare memory AND not busy computing.

    Memory alone is not enough: a job can be between allocations and still
    be about to grab everything back.
    """
    return gpu.free_mib >= min_free_mib and gpu.util_pct <= max_util_pct


def send_email(subject: str, body: str, to_addr: str, user: str, password: str) -> None:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to_addr
    msg.set_content(body)

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as smtp:
        smtp.starttls()
        smtp.login(user, password)
        smtp.send_message(msg)


def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}", flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ONE-TIME SETUP (Gmail blocks normal passwords, so you need an App Password):

  1. Turn on 2-Step Verification:  https://myaccount.google.com/security
  2. Create an App Password:       https://myaccount.google.com/apppasswords
     Pick "Mail" / "Other", name it "gpu watch". Google shows you a
     16-character code -- that is the password below.
  3. Put it in your shell (and in ~/.bashrc so it survives a reboot):

       export GPU_WATCH_SMTP_USER="you@gmail.com"
       export GPU_WATCH_SMTP_PASS="the16charcode"

  4. Check it works:   env/bin/python scripts/gpu_watch.py --test-email
""",
    )
    p.add_argument("--min-free-mib", type=int, default=40000,
                   help="Consider the GPU free at this much spare memory (default: 40000).")
    p.add_argument("--max-util", type=int, default=20,
                   help="...and at or below this %% utilisation (default: 20).")
    p.add_argument("--gpu", type=int, default=None,
                   help="Watch only this GPU index (default: any GPU).")
    p.add_argument("--interval", type=int, default=60,
                   help="Seconds between checks (default: 60).")
    p.add_argument("--confirmations", type=int, default=3,
                   help="Consecutive free checks required before emailing (default: 3). "
                        "Stops a brief gap between two jobs triggering a false alarm.")
    p.add_argument("--to", default=os.environ.get("GPU_WATCH_TO"),
                   help=f"Recipient (default: $GPU_WATCH_TO, else {DEFAULT_TO}).")
    p.add_argument("--keep-going", action="store_true",
                   help="Keep watching after the first email instead of exiting. "
                        "Re-arms once the GPU is busy again.")
    p.add_argument("--timeout-hours", type=float, default=None,
                   help="Give up after this long (default: watch forever).")
    p.add_argument("--test-email", action="store_true",
                   help="Send one test email now and exit -- use this to check your setup.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the email instead of sending it. No credentials needed.")
    return p.parse_args(argv)


def credentials(dry_run: bool):
    user = os.environ.get("GPU_WATCH_SMTP_USER")
    password = os.environ.get("GPU_WATCH_SMTP_PASS")
    if dry_run:
        return user or "(dry-run)", password or "(dry-run)"
    if not user or not password:
        sys.exit(
            "Missing email credentials.\n\n"
            "  export GPU_WATCH_SMTP_USER=\"you@gmail.com\"\n"
            "  export GPU_WATCH_SMTP_PASS=\"your-16-char-app-password\"\n\n"
            "Run with --help for how to create the App Password, or use --dry-run "
            "to test the GPU watching without sending anything."
        )
    return user, password


def main(argv=None) -> int:
    args = parse_args(argv)
    user, password = credentials(args.dry_run)
    to_addr = args.to or DEFAULT_TO
    host = socket.gethostname()

    def deliver(subject: str, body: str) -> None:
        if args.dry_run:
            log(f"[dry-run] would email {to_addr}\n--- {subject} ---\n{body}\n---")
            return
        send_email(subject, body, to_addr, user, password)
        log(f"Emailed {to_addr}: {subject}")

    if args.test_email:
        deliver(
            f"[gpu-watch] test from {host}",
            "This is a test from scripts/gpu_watch.py.\n\n"
            "If you are reading this, the email setup works and the watcher "
            "will be able to reach you when the GPU frees up.\n\n"
            f"Current state:\n" + "\n".join(f"  {g}" for g in query_gpus()),
        )
        return 0

    target = "any GPU" if args.gpu is None else f"GPU {args.gpu}"
    log(f"Watching {target} on {host}: free = >={args.min_free_mib:,} MiB and <={args.max_util}% util")
    log(f"Checking every {args.interval}s, {args.confirmations} consecutive checks required. "
        f"Will email {to_addr}.")

    deadline = time.time() + args.timeout_hours * 3600 if args.timeout_hours else None
    streak = 0
    armed = True          # False after an email, until the GPU is busy again
    consecutive_errors = 0

    while True:
        if deadline and time.time() > deadline:
            log(f"Timed out after {args.timeout_hours}h without the GPU freeing up. Exiting.")
            return 1

        try:
            gpus = query_gpus()
            consecutive_errors = 0
        except RuntimeError as exc:
            consecutive_errors += 1
            log(f"WARNING: {exc} (error {consecutive_errors})")
            # Transient nvidia-smi failures are common under heavy load;
            # only give up if it keeps failing.
            if consecutive_errors >= 10:
                log("nvidia-smi failed 10 times in a row. Exiting.")
                return 1
            time.sleep(args.interval)
            continue

        watched = [g for g in gpus if args.gpu is None or g.index == args.gpu]
        if not watched:
            log(f"ERROR: GPU {args.gpu} does not exist. Found: {[g.index for g in gpus]}")
            return 1

        free_now = [g for g in watched if is_free(g, args.min_free_mib, args.max_util)]

        if free_now and armed:
            streak += 1
            log(f"Free ({streak}/{args.confirmations}): " + "; ".join(str(g) for g in free_now))
            if streak >= args.confirmations:
                body = (
                    f"The GPU on {host} is free.\n\n"
                    + "\n".join(f"  {g}" for g in free_now)
                    + "\n\nAll GPUs:\n"
                    + "\n".join(f"  {g}" for g in gpus)
                    + "\n\nProcesses still on the GPU:\n  "
                    + running_processes().replace("\n", "\n  ")
                    + f"\n\nFree for {streak} consecutive checks "
                      f"({streak * args.interval}s).\n"
                      f"Watched by scripts/gpu_watch.py on {host}.\n"
                )
                deliver(f"[gpu-watch] GPU free on {host}", body)
                if not args.keep_going:
                    return 0
                armed = False
                streak = 0
                log("Re-arming: will email again after the GPU gets busy and frees up once more.")
        else:
            if streak:
                log("GPU busy again -- streak reset.")
            streak = 0
            if not armed and not free_now:
                armed = True
                log("GPU busy again -- re-armed.")
            busiest = max(watched, key=lambda g: g.used_mib)
            log(f"Busy: {busiest}")

        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Stopped.")
        raise SystemExit(130)
