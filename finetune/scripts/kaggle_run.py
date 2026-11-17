"""Queue runner for Kaggle notebooks of the fine-tuning study (laptop side).

Not to be confused with scripts/kaggle_run.py, the v1.0 stage runner that executes INSIDE a notebook; that
file is frozen with v1.0, which is why this one lives under finetune/.

  .venv/bin/python finetune/scripts/kaggle_run.py add --id ft_test --notebook finetune/notebooks/x.ipynb \
        --max-minutes 10 --note "why"
  .venv/bin/python finetune/scripts/kaggle_run.py launch [--dry-run]     # at most one launch per call
  .venv/bin/python finetune/scripts/kaggle_run.py poll   [--dry-run]     # status; download + STATUS.txt check when done
  .venv/bin/python finetune/scripts/kaggle_run.py status                 # the queue and the GPU-hour ledger, no network
  caffeinate -i .venv/bin/python finetune/scripts/kaggle_run.py run --every 300   # loop: poll, then launch whatever the
        # safeguards allow; ends when nothing is in flight and nothing more can be launched (queue empty, HALT, cap or
        # quota), or after --max-hours. A refusal is a reason to stop launching, never to retry the same id.
  .venv/bin/python finetune/scripts/kaggle_run.py external --hours 3 --note "launched by hand in the web editor"
  .venv/bin/python finetune/scripts/kaggle_run.py clear-halt --reason "..."   # the owner's decision, never automatic

Safeguards (each has a test in finetune/tests/test_kaggle_run.py):
  1. Queue file. Nothing is launched that is not an entry of finetune/kaggle/queue.json, and entries are
     launched in order. Every state change is written to the file before the next action.
  2. At most MAX_CONCURRENT (2) runs in flight (state launched/running).
  3. Weekly GPU-hour cap (26 h per Kaggle quota week, which starts Saturday 00:00 UTC). Before a launch the
     runner adds up: hours charged to finished runs this week + the FULL time limit of every run in flight +
     the full time limit of the candidate + hours entered with `external`. If that exceeds the cap it refuses.
     The time limit is also passed to Kaggle (--timeout), so a run cannot outlive what was reserved for it.
     Hours are session wall-clock hours (how Kaggle charges a session, whether it has one T4 or two).
     Second limit: `quota --hours-left H` records what Kaggle's own page says is left this week. From then until the
     quota week ends, the hours of runs launched after that moment (charged, or full limits while in flight) plus
     the candidate's limit may not exceed H. The cap is the owner's rule; this is Kaggle's.
  4. No automatic relaunch. An id is launched once in its life; there is no retry and no requeue command.
     Running something again means a new `add` with a new id, typed by a person or by the agent on instruction.
  5. Stop and report on any failure. A run that errors, is cancelled, times out, cannot be downloaded, or whose
     STATUS.txt is missing or does not start with OK, is marked failed and the file finetune/kaggle/HALT is
     written. While HALT exists nothing is launched. Runs already in flight are left alone and still polled.
  6. --dry-run on launch and poll: prints the exact commands and the ledger arithmetic; pushes nothing,
     downloads nothing, writes nothing.
  7. Only the Kaggle CLI inside this checkout's .venv is called; anything else is refused. Kernels are
     always pushed private. The notebook that is pushed is hashed, and the hash and commit are recorded.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KDIR = ROOT / "finetune" / "kaggle"
MAX_CONCURRENT = 2
WEEKLY_CAP_HOURS = 26.0             # the owner's cap (raised from 20 on 16 Nov)
QUEUE_PATIENCE_MIN = 180              # a run Kaggle has not started after this long is reported as a failure
ACCELERATOR = "NvidiaTeslaT4"            # the CLI's name for "GPU T4 x2"
IN_FLIGHT = ("launched", "running")
OK_STATES = {"complete"}
BAD_STATES = {"error", "cancelrequested", "cancelacknowledged", "cancelled", "canceled", "timeout", "failed"}
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_]{2,40}$")


class Refused(Exception):
    """A safeguard said no. The message is the report."""


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def week_start(t: dt.datetime) -> dt.datetime:
    """Start of the Kaggle quota week containing t: the most recent Saturday 00:00 UTC."""
    d = t.astimezone(dt.timezone.utc)
    back = (d.weekday() - 5) % 7          # Monday = 0, Saturday = 5
    return (d - dt.timedelta(days=back)).replace(hour=0, minute=0, second=0, microsecond=0)


class Runner:
    def __init__(self, kdir: Path = KDIR, kaggle: Path | None = None, clock=now, enforce_venv: bool = True):
        self.kdir, self.clock = Path(kdir), clock
        self.kaggle = Path(kaggle) if kaggle else ROOT / ".venv" / "bin" / "kaggle"
        if enforce_venv and (self.kaggle.parent.parent.name != ".venv" or ROOT not in self.kaggle.parents):
            raise Refused(f"refusing to call {self.kaggle}: only the Kaggle CLI in {ROOT / '.venv'} is used")
        self.qpath, self.halt = self.kdir / "queue.json", self.kdir / "HALT"

    # ------------------------------------------------------------------ state
    def load(self) -> dict:
        if not self.qpath.exists():
            return {"runs": [], "external": []}
        return json.loads(self.qpath.read_text())

    def save(self, q: dict) -> None:
        self.kdir.mkdir(parents=True, exist_ok=True)
        tmp = self.qpath.with_suffix(".tmp")
        tmp.write_text(json.dumps(q, indent=1) + "\n")
        tmp.replace(self.qpath)

    def lock(self):
        self.kdir.mkdir(parents=True, exist_ok=True)
        f = open(self.kdir / ".lock", "w")
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Refused("another kaggle_run.py is working on the queue; nothing done")
        return f

    def cli(self, *args: str, timeout: int = 900) -> subprocess.CompletedProcess:
        if not self.kaggle.exists():
            raise Refused(f"{self.kaggle} does not exist")
        return subprocess.run([str(self.kaggle), *args], capture_output=True, text=True, timeout=timeout)

    # ------------------------------------------------------------------ ledger
    def ledger(self, q: dict) -> dict:
        t = self.clock()
        w0 = week_start(t)
        charged = reserved = 0.0
        for r in q["runs"]:
            if not r.get("launched_at") or parse(r["launched_at"]) < w0:
                continue
            if r["state"] in IN_FLIGHT:
                reserved += r["max_minutes"] / 60
            else:
                charged += r.get("gpu_hours_charged", r["max_minutes"] / 60)
        external = sum(e["hours"] for e in q.get("external", []) if parse(e["at"]) >= w0)
        led = {"week_start": iso(w0), "charged": round(charged, 3), "reserved_in_flight": round(reserved, 3),
               "external": round(external, 3), "total": round(charged + reserved + external, 3),
               "cap": WEEKLY_CAP_HOURS, "kaggle_left": None}
        qs = [x for x in q.get("quota", []) if parse(x["at"]) >= w0]
        if qs:
            last = qs[-1]
            since = sum((r["max_minutes"] / 60 if r["state"] in IN_FLIGHT else r.get("gpu_hours_charged", r["max_minutes"] / 60))
                        for r in q["runs"] if r.get("launched_at") and parse(r["launched_at"]) >= parse(last["at"]) and r.get("gpu", True))
            led["kaggle_left"] = round(last["hours_left"] - since, 3)
            led["kaggle_left_recorded"] = f"{last['hours_left']} h at {last['at']}"
        return led

    # ------------------------------------------------------------------ commands
    def add(self, rid: str, notebook: str, max_minutes: float, note: str = "", owner: str | None = None,
            expect_gpu: bool = True) -> dict:
        if not ID_RE.match(rid):
            raise Refused(f"id {rid!r}: use lower-case letters, digits and underscores (3 to 41 characters)")
        nb = (ROOT / notebook).resolve()
        if not nb.is_file() or nb.suffix != ".ipynb" or ROOT not in nb.parents:
            raise Refused(f"notebook {notebook} is not an .ipynb file inside {ROOT}")
        if not 1 <= max_minutes <= 12 * 60:
            raise Refused("max-minutes must be between 1 and 720 (Kaggle ends a session at 12 hours)")
        q = self.load()
        if any(r["id"] == rid for r in q["runs"]):
            raise Refused(f"id {rid} is already in the queue file; an id is used once (no relaunch). Choose a new id")
        owner = owner or self.username()
        r = {"id": rid, "notebook": str(nb.relative_to(ROOT)), "kernel": f"{owner}/{rid.replace('_', '-')}",
             "max_minutes": max_minutes, "gpu": expect_gpu, "note": note, "state": "queued", "added_at": iso(self.clock())}
        q["runs"].append(r)
        self.save(q)
        return r

    def username(self) -> str:
        out = self.cli("config", "view").stdout
        m = re.search(r"username:\s*(\S+)", out)
        if not m:
            raise Refused("could not read the Kaggle user name from `kaggle config view`; pass --owner")
        return m.group(1)

    def external(self, hours: float, note: str) -> None:
        q = self.load()
        q.setdefault("external", []).append({"hours": hours, "note": note, "at": iso(self.clock())})
        self.save(q)

    def quota(self, hours_left: float, note: str = "") -> None:
        q = self.load()
        q.setdefault("quota", []).append({"hours_left": hours_left, "note": note, "at": iso(self.clock())})
        self.save(q)

    def launch(self, dry: bool = False, say=print) -> dict | None:
        """Launch the first queued run if every safeguard allows it. At most one launch per call."""
        q = self.load()
        if self.halt.exists():
            raise Refused("HALT is set, nothing is launched:\n" + self.halt.read_text().rstrip())
        nxt = next((r for r in q["runs"] if r["state"] == "queued"), None)
        if nxt is None:
            say("queue: nothing queued")
            return None
        flying = [r["id"] for r in q["runs"] if r["state"] in IN_FLIGHT]
        if len(flying) >= MAX_CONCURRENT:
            raise Refused(f"{len(flying)} runs in flight ({', '.join(flying)}); the limit is {MAX_CONCURRENT}. "
                          f"{nxt['id']} stays queued")
        led = self.ledger(q)
        need = nxt["max_minutes"] / 60 if nxt["gpu"] else 0.0
        say(f"ledger for the week from {led['week_start']}: charged {led['charged']} h + in flight (full time limits) "
            f"{led['reserved_in_flight']} h + external {led['external']} h + this run's limit {need:.2f} h "
            f"= {led['total'] + need:.2f} h of {led['cap']} h")
        if led["total"] + need > WEEKLY_CAP_HOURS + 1e-9:
            raise Refused(f"launching {nxt['id']} could bring the week to {led['total'] + need:.2f} GPU hours; "
                          f"the cap is {WEEKLY_CAP_HOURS}. It stays queued")
        if led["kaggle_left"] is not None:
            say(f"Kaggle's own quota: {led['kaggle_left_recorded']}; left after runs launched since: {led['kaggle_left']} h")
            if need > led["kaggle_left"] + 1e-9:
                raise Refused(f"{nxt['id']} has a limit of {need:.2f} h and only {led['kaggle_left']} h of Kaggle's weekly quota "
                              f"are left by the record of {led['kaggle_left_recorded']}. It stays queued")
        nb = ROOT / nxt["notebook"]
        if not nb.is_file():
            raise Refused(f"{nxt['notebook']} no longer exists; {nxt['id']} stays queued")
        sha = hashlib.sha256(nb.read_bytes()).hexdigest()
        push = self.kdir / "push" / nxt["id"]
        meta = {"id": nxt["kernel"], "title": nxt["kernel"].split("/")[1], "code_file": nb.name, "language": "python",
                "kernel_type": "notebook", "is_private": "true", "enable_gpu": "true" if nxt["gpu"] else "false",
                "enable_internet": "true", "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
                "model_sources": []}
        cmd = ["kernels", "push", "-p", str(push), "-t", str(int(nxt["max_minutes"] * 60))]
        if nxt["gpu"]:
            cmd += ["--accelerator", ACCELERATOR]
        if dry:
            say(f"[dry run] would write {push}/kernel-metadata.json: {json.dumps(meta)}")
            say(f"[dry run] would run: {self.kaggle} {' '.join(cmd)}")
            say(f"[dry run] notebook {nxt['notebook']} sha256 {sha[:16]}; nothing pushed, queue file unchanged")
            return None
        if push.exists():
            shutil.rmtree(push)
        push.mkdir(parents=True)
        shutil.copy2(nb, push / nb.name)
        (push / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        # the state is written BEFORE the push: if this process dies mid-push the id is spent, not launched twice
        nxt.update(state="launched", launched_at=iso(self.clock()), notebook_sha256=sha, commit=self.commit())
        self.save(q)
        p = self.cli(*cmd)
        nxt["push_output"] = (p.stdout + p.stderr).strip()[-600:]
        if p.returncode != 0 or "error" in nxt["push_output"].lower():
            self.fail(q, nxt, "the push was rejected: " + nxt["push_output"][-300:], charged=0.0)
            raise Refused(f"push of {nxt['id']} failed; HALT written. Output:\n{nxt['push_output']}")
        self.save(q)
        say(f"launched {nxt['id']} as {nxt['kernel']} (limit {nxt['max_minutes']:.0f} min): {nxt['push_output']}")
        return nxt

    def commit(self) -> str:
        p = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return p.stdout.strip() + ("+uncommitted" if dirty else "")

    def fail(self, q: dict, r: dict, why: str, charged: float | None = None) -> None:
        r.update(state="failed", failure=why, finished_at=iso(self.clock()))
        if charged is not None:
            r["gpu_hours_charged"] = charged
        self.save(q)
        self.kdir.mkdir(parents=True, exist_ok=True)
        with open(self.halt, "a") as f:
            f.write(f"{iso(self.clock())}  {r['id']} ({r['kernel']}): {why}\n")

    def poll(self, dry: bool = False, say=print) -> list[dict]:
        """Ask Kaggle about every run in flight. Finished: download, check STATUS.txt. Never launches anything."""
        q = self.load()
        changed = []
        for r in [r for r in q["runs"] if r["state"] in IN_FLIGHT]:
            if dry:
                say(f"[dry run] would run: {self.kaggle} kernels status {r['kernel']}; if complete: kernels output "
                    f"{r['kernel']} -p {self.kdir / 'downloads' / r['id']}, then check STATUS.txt")
                continue
            p = self.cli("kernels", "status", r["kernel"])
            m = re.search(r'has status "(?:KernelWorkerStatus\.)?(\w+)"', p.stdout + p.stderr)
            st = m.group(1).lower().replace("_", "") if m else None
            mins = (self.clock() - parse(r["launched_at"])).total_seconds() / 60
            if st == "running" and not r.get("running_since"):
                r["running_since"] = iso(self.clock())
            # time on the GPU: from the first poll that saw it running (Kaggle's own queue can hold a run for a long
            # time and is not charged). A run never seen running is charged from launch: an upper bound.
            ran = (self.clock() - parse(r.get("running_since") or r["launched_at"])).total_seconds() / 60
            used = round(min(ran, r["max_minutes"]) / 60, 3) if r["gpu"] else 0.0
            r["last_status"], r["last_polled"] = st or (p.stdout + p.stderr).strip()[-200:], iso(self.clock())
            if st in OK_STATES:
                r["gpu_hours_charged"] = used
                why = self.collect(r)
                if why:
                    self.fail(q, r, why)
                else:
                    r.update(state="complete", finished_at=iso(self.clock()))
                changed.append(r)
            elif st in BAD_STATES:
                self.collect(r)                                  # keep whatever there is; the run has failed either way
                self.fail(q, r, f"Kaggle reports status {st}", charged=used)
                changed.append(r)
            elif st is None:
                say(f"{r['id']}: status could not be read ({r['last_status']}); left as {r['state']}")
            else:
                if r.get("running_since") and ran > r["max_minutes"] + 15:
                    self.fail(q, r, f"still '{st}' {ran:.0f} min after it started, limit {r['max_minutes']:.0f} min; check it "
                                    f"in the browser", charged=r["max_minutes"] / 60 if r["gpu"] else 0.0)
                    changed.append(r)
                elif not r.get("running_since") and mins > QUEUE_PATIENCE_MIN:
                    self.fail(q, r, f"still '{st}' {mins:.0f} min after launch without having started; check it in the "
                                    f"browser (it may still start: its full limit stays counted)",
                              charged=r["max_minutes"] / 60 if r["gpu"] else 0.0)
                    changed.append(r)
                else:
                    r["state"] = "running" if st == "running" else r["state"]
            self.save(q)
            say(f"{r['id']}: {r['state']} (Kaggle: {r['last_status']}; {mins:.0f} min since launch)"
                + (f"  FAILURE: {r['failure']}" if r["state"] == "failed" else ""))
        if self.halt.exists() and not dry:
            say("HALT is set; nothing will be launched until the owner clears it:\n" + self.halt.read_text().rstrip())
        return changed

    def collect(self, r: dict) -> str:
        """Download the output and the log of a finished run. Returns '' if STATUS.txt says OK, else the reason."""
        dest = self.kdir / "downloads" / r["id"]
        dest.mkdir(parents=True, exist_ok=True)
        p = self.cli("kernels", "output", r["kernel"], "-p", str(dest), "-o", timeout=3600)
        r["download_output"] = (p.stdout + p.stderr).strip()[-300:]
        try:
            lg = self.cli("kernels", "logs", r["kernel"], timeout=300)
            (dest / "kaggle_log.txt").write_text(lg.stdout + lg.stderr)
        except Exception as e:                       # the log is a convenience; its absence is not the verdict
            r["log_error"] = f"{type(e).__name__}: {e}"[:200]
        files = sorted(str(f.relative_to(dest)) for f in dest.rglob("*") if f.is_file())
        r["downloaded"] = {f: (dest / f).stat().st_size for f in files}
        if p.returncode != 0:
            return "download failed: " + r["download_output"]
        status = [f for f in dest.rglob("STATUS.txt")]
        if len(status) != 1:
            return f"expected exactly one STATUS.txt in the download, found {len(status)}"
        lines = status[0].read_text().splitlines()
        first = lines[0].strip() if lines else ""
        r["status_txt"] = first
        if first != "OK":
            return f"STATUS.txt says {first!r}, not OK"
        return ""

    def clear_halt(self, reason: str) -> None:
        if not self.halt.exists():
            raise Refused("HALT is not set")
        log = self.kdir / "halt_history.txt"
        with open(log, "a") as f:
            f.write(self.halt.read_text() + f"{iso(self.clock())}  CLEARED: {reason}\n\n")
        self.halt.unlink()

    def run_loop(self, every: int = 300, max_hours: float = 30.0, say=print, sleep=None) -> str:
        import time as _t
        sleep = sleep or _t.sleep
        t_end = _t.time() + max_hours * 3600
        while True:
            self.poll(say=say)
            launched_or_refused = None
            while True:                                   # launch as many as the safeguards allow, one call at a time
                try:
                    r = self.launch(say=say)
                except Refused as x:
                    launched_or_refused = str(x); break
                if r is None:
                    break
            q = self.load()
            flying = [r for r in q["runs"] if r["state"] in IN_FLIGHT]
            if not flying:
                why = "HALT is set" if self.halt.exists() else (launched_or_refused or "nothing queued")
                say(f"loop ends: nothing in flight; {why}")
                return why
            if _t.time() > t_end:
                say("loop ends: --max-hours reached with runs in flight; run the loop again to keep polling")
                return "max hours"
            sleep(every)

    def report(self, say=print) -> None:
        q = self.load()
        led = self.ledger(q)
        say(f"GPU hours, week from {led['week_start']}: charged {led['charged']} + in flight {led['reserved_in_flight']} "
            f"+ external {led['external']} = {led['total']} of {led['cap']}")
        if led["kaggle_left"] is not None:
            say(f"Kaggle's own quota: {led['kaggle_left']} h left ({led['kaggle_left_recorded']}, minus runs launched since)")
        say("HALT: " + (self.halt.read_text().rstrip() if self.halt.exists() else "not set"))
        for r in q["runs"]:
            say(f"  {r['state']:9s} {r['id']:28s} limit {r['max_minutes']:5.0f} min  "
                f"charged {r.get('gpu_hours_charged', '-')!s:6s} {r['kernel']}  {r.get('failure', '')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add"); a.add_argument("--id", required=True); a.add_argument("--notebook", required=True)
    a.add_argument("--max-minutes", type=float, required=True); a.add_argument("--note", default="")
    a.add_argument("--owner", default=None); a.add_argument("--cpu", action="store_true")
    for name in ("launch", "poll"):
        s = sub.add_parser(name); s.add_argument("--dry-run", action="store_true")
    sub.add_parser("status")
    rl = sub.add_parser("run"); rl.add_argument("--every", type=int, default=300); rl.add_argument("--max-hours", type=float, default=30.0)
    e = sub.add_parser("external"); e.add_argument("--hours", type=float, required=True); e.add_argument("--note", required=True)
    k = sub.add_parser("quota"); k.add_argument("--hours-left", type=float, required=True); k.add_argument("--note", default="")
    c = sub.add_parser("clear-halt"); c.add_argument("--reason", required=True)
    args = ap.parse_args(argv)
    try:
        R = Runner()
        lock = R.lock()                     # noqa: F841  (held until exit)
        if args.cmd == "add":
            r = R.add(args.id, args.notebook, args.max_minutes, args.note, args.owner, not args.cpu)
            print(f"queued {r['id']} -> {r['kernel']} (limit {r['max_minutes']:.0f} min)")
        elif args.cmd == "launch":
            R.launch(dry=args.dry_run)
        elif args.cmd == "poll":
            R.poll(dry=args.dry_run)
        elif args.cmd == "run":
            R.run_loop(args.every, args.max_hours)
        elif args.cmd == "status":
            R.report()
        elif args.cmd == "external":
            R.external(args.hours, args.note)
        elif args.cmd == "quota":
            R.quota(args.hours_left, args.note)
        elif args.cmd == "clear-halt":
            R.clear_halt(args.reason)
    except Refused as x:
        print("REFUSED: " + str(x), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
