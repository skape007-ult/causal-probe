"""Safeguards of finetune/scripts/kaggle_run.py, against a fake Kaggle CLI (no network)."""
import datetime as dt
import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("kaggle_run_ft", ROOT / "finetune" / "scripts" / "kaggle_run.py")
K = importlib.util.module_from_spec(spec); sys.modules["kaggle_run_ft"] = K; spec.loader.exec_module(K)

FAKE = r'''#!/bin/sh
# fake kaggle: logs every call; behaviour from files next to it
D="$(dirname "$0")"
echo "$@" >> "$D/calls.log"
case "$1 $2" in
  "config view") echo "- username: tester" ;;
  "kernels push") if [ -f "$D/push_fails" ]; then echo "Kernel push error: nope"; exit 1; fi
                  echo "Kernel version 1 successfully pushed." ;;
  "kernels status") echo "$3 has status \"KernelWorkerStatus.$(cat "$D/status")\"" ;;
  "kernels output") if [ -f "$D/status_txt" ]; then mkdir -p "$5/kaggle_out"; cp "$D/status_txt" "$5/kaggle_out/STATUS.txt"; fi ;;
  "kernels logs") echo "log line" ;;
esac
'''


class Clock:
    def __init__(self):
        self.t = dt.datetime(2026, 11, 16, 12, 0, tzinfo=dt.timezone.utc)      # a Monday

    def __call__(self):
        return self.t

    def add(self, **kw):
        self.t += dt.timedelta(**kw)


@pytest.fixture
def env(tmp_path, monkeypatch):
    fake = tmp_path / "bin" / "kaggle"
    fake.parent.mkdir()
    fake.write_text(FAKE); fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    (fake.parent / "status").write_text("RUNNING")
    (fake.parent / "status_txt").write_text("OK\nrun: x\n")
    nbdir = ROOT / "finetune" / "notebooks"
    nb = next(nbdir.glob("*.ipynb"))
    clock = Clock()
    R = K.Runner(kdir=tmp_path / "k", kaggle=fake, clock=clock, enforce_venv=False)
    R.nb, R.bin, R.clk = str(nb.relative_to(ROOT)), fake.parent, clock
    return R


def calls(R):
    f = R.bin / "calls.log"
    return f.read_text().splitlines() if f.exists() else []


def pushes(R):
    return [c for c in calls(R) if c.startswith("kernels push")]


def test_only_the_venv_cli_is_accepted(tmp_path):
    with pytest.raises(K.Refused):
        K.Runner(kdir=tmp_path, kaggle=Path("/usr/local/bin/kaggle"))
    assert K.Runner(kdir=tmp_path).kaggle == ROOT / ".venv" / "bin" / "kaggle"


def test_launch_is_private_gpu_t4_with_time_limit(env):
    env.add("run_a", env.nb, 10)
    env.launch(say=lambda *_: None)
    assert pushes(env) == [f"kernels push -p {env.kdir / 'push' / 'run_a'} -t 600 --accelerator NvidiaTeslaT4"]
    meta = json.loads((env.kdir / "push" / "run_a" / "kernel-metadata.json").read_text())
    assert meta["is_private"] == "true" and meta["enable_gpu"] == "true" and meta["enable_internet"] == "true"
    assert meta["id"] == "tester/run-a"
    assert env.load()["runs"][0]["state"] == "launched"


def test_dry_run_changes_nothing(env):
    env.add("run_a", env.nb, 10)
    before = env.qpath.read_text()
    out = []
    env.launch(dry=True, say=out.append)
    assert pushes(env) == [] and env.qpath.read_text() == before and not (env.kdir / "push").exists()
    assert any("would run" in o and "kernels push" in o for o in out)
    env.launch(say=lambda *_: None)
    n = len(calls(env))
    env.poll(dry=True, say=lambda *_: None)
    assert len(calls(env)) == n


def test_nothing_outside_the_queue_and_one_launch_per_call(env):
    assert env.launch(say=lambda *_: None) is None and pushes(env) == []
    env.add("run_a", env.nb, 10); env.add("run_b", env.nb, 10)
    env.launch(say=lambda *_: None)
    assert len(pushes(env)) == 1


def test_max_two_concurrent(env):
    for i in "abc":
        env.add(f"run_{i}", env.nb, 10)
    env.launch(say=lambda *_: None); env.launch(say=lambda *_: None)
    with pytest.raises(K.Refused, match="limit is 2"):
        env.launch(say=lambda *_: None)
    assert len(pushes(env)) == 2 and env.load()["runs"][2]["state"] == "queued"
    (env.bin / "status").write_text("COMPLETE")          # both finish; the third may go
    env.clk.add(minutes=5); env.poll(say=lambda *_: None)
    env.launch(say=lambda *_: None)
    assert len(pushes(env)) == 3


def test_weekly_cap_counts_full_limits_of_runs_in_flight(env):
    env.add("run_a", env.nb, 720); env.add("run_b", env.nb, 720); env.add("run_c", env.nb, 121)
    env.launch(say=lambda *_: None); env.launch(say=lambda *_: None)      # 12 h + 12 h reserved
    (env.bin / "status").write_text("RUNNING"); env.clk.add(minutes=1); env.poll(say=lambda *_: None)
    (env.bin / "status").write_text("COMPLETE")
    env.clk.add(hours=12); env.poll(say=lambda *_: None)                   # charged 24 h; 2.02 h more would pass 26
    with pytest.raises(K.Refused, match="cap is 26"):
        env.launch(say=lambda *_: None)
    assert len(pushes(env)) == 2
    env.clk.add(days=5)                                                    # Saturday 22:00 UTC: a new quota week
    env.launch(say=lambda *_: None)
    assert len(pushes(env)) == 3


def test_cap_reserves_before_anything_is_charged(env):
    env.external(3.0, "by hand")
    env.add("run_a", env.nb, 720); env.add("run_b", env.nb, 720)
    env.launch(say=lambda *_: None)
    with pytest.raises(K.Refused, match="cap is 26"):                      # 3 h + 12 h in flight + 12 h > 26
        env.launch(say=lambda *_: None)


def test_external_hours_count(env):
    env.external(25.95, "by hand")
    env.add("run_a", env.nb, 10)
    with pytest.raises(K.Refused, match="cap is 26"):
        env.launch(say=lambda *_: None)


def test_week_starts_saturday_utc():
    f = lambda *a: K.week_start(dt.datetime(*a, tzinfo=dt.timezone.utc))
    assert f(2026, 11, 16, 12) == dt.datetime(2026, 11, 14, tzinfo=dt.timezone.utc)
    assert f(2026, 11, 14, 0) == dt.datetime(2026, 11, 14, tzinfo=dt.timezone.utc)
    assert f(2026, 11, 13, 23, 59) == dt.datetime(2026, 11, 7, tzinfo=dt.timezone.utc)


def test_no_relaunch_of_an_id(env):
    env.add("run_a", env.nb, 10)
    env.launch(say=lambda *_: None)
    with pytest.raises(K.Refused, match="used once"):
        env.add("run_a", env.nb, 10)
    (env.bin / "status").write_text("ERROR")
    env.poll(say=lambda *_: None)
    assert env.load()["runs"][0]["state"] == "failed"
    env.clear_halt("test")
    assert env.launch(say=lambda *_: None) is None and len(pushes(env)) == 1      # failed is not queued again


@pytest.mark.parametrize("kaggle_status,status_txt,why", [
    ("ERROR", "OK\n", "status error"),
    ("CANCEL_ACKNOWLEDGED", "OK\n", "status cancel"),
    ("COMPLETE", "FAILED\nproblems:\n", "says 'failed'"),
    ("COMPLETE", "SKIPPED\n", "says 'skipped'"),
    ("COMPLETE", None, "found 0"),
])
def test_any_failure_halts_the_queue(env, kaggle_status, status_txt, why):
    env.add("run_a", env.nb, 10); env.add("run_b", env.nb, 10)
    env.launch(say=lambda *_: None)
    (env.bin / "status").write_text(kaggle_status)
    if status_txt is None:
        (env.bin / "status_txt").unlink()
    else:
        (env.bin / "status_txt").write_text(status_txt)
    env.poll(say=lambda *_: None)
    r = env.load()["runs"][0]
    assert r["state"] == "failed" and why in r["failure"].lower()
    assert env.halt.exists()
    with pytest.raises(K.Refused, match="HALT"):
        env.launch(say=lambda *_: None)
    assert len(pushes(env)) == 1
    env.clear_halt("looked at it")
    assert not env.halt.exists() and "CLEARED: looked at it" in (env.kdir / "halt_history.txt").read_text()


def test_rejected_push_halts_and_spends_the_id(env):
    (env.bin / "push_fails").write_text("")
    env.add("run_a", env.nb, 10)
    with pytest.raises(K.Refused, match="push of run_a failed"):
        env.launch(say=lambda *_: None)
    assert env.halt.exists() and env.load()["runs"][0]["state"] == "failed"


def test_overdue_run_is_reported_not_relaunched(env):
    env.add("run_a", env.nb, 10)
    env.launch(say=lambda *_: None)
    (env.bin / "status").write_text("QUEUED")
    env.clk.add(minutes=90); env.poll(say=lambda *_: None)                 # Kaggle's own queue: waiting is not a failure
    assert env.load()["runs"][0]["state"] == "launched" and not env.halt.exists()
    (env.bin / "status").write_text("RUNNING")
    env.clk.add(minutes=1); env.poll(say=lambda *_: None)
    env.clk.add(minutes=20); env.poll(say=lambda *_: None)
    assert env.load()["runs"][0]["state"] == "running"
    env.clk.add(minutes=10); env.poll(say=lambda *_: None)                 # 30 min after it started, limit 10
    assert env.load()["runs"][0]["state"] == "failed" and env.halt.exists() and len(pushes(env)) == 1


def test_never_started_is_reported(env):
    env.add("run_a", env.nb, 10)
    env.launch(say=lambda *_: None)
    (env.bin / "status").write_text("QUEUED")
    env.clk.add(minutes=181); env.poll(say=lambda *_: None)
    assert env.load()["runs"][0]["state"] == "failed" and env.halt.exists()


def test_charge_counts_from_running_not_from_kaggles_queue(env):
    env.add("run_a", env.nb, 10)
    env.launch(say=lambda *_: None)
    (env.bin / "status").write_text("QUEUED"); env.clk.add(minutes=60); env.poll(say=lambda *_: None)
    (env.bin / "status").write_text("RUNNING"); env.clk.add(minutes=1); env.poll(say=lambda *_: None)
    (env.bin / "status").write_text("COMPLETE"); env.clk.add(minutes=3); env.poll(say=lambda *_: None)
    assert env.load()["runs"][0]["gpu_hours_charged"] == 0.05


def test_complete_run_is_downloaded_and_charged(env):
    env.add("run_a", env.nb, 10)
    env.launch(say=lambda *_: None)
    (env.bin / "status").write_text("COMPLETE")
    env.clk.add(minutes=6); env.poll(say=lambda *_: None)
    r = env.load()["runs"][0]
    assert r["state"] == "complete" and r["status_txt"] == "OK" and r["gpu_hours_charged"] == 0.1
    assert "kaggle_out/STATUS.txt" in r["downloaded"] and not env.halt.exists()


def test_kaggles_own_quota_left_is_a_second_limit(env):
    env.quota(1.0, "from the Kaggle page")
    env.add("run_a", env.nb, 45); env.add("run_b", env.nb, 30); env.add("run_c", env.nb, 10)
    env.launch(say=lambda *_: None)                                        # 0.75 h of 1.0
    with pytest.raises(K.Refused, match="weekly quota"):
        env.launch(say=lambda *_: None)                                    # + 0.5 h would pass what Kaggle has left
    (env.bin / "status").write_text("RUNNING"); env.clk.add(minutes=1); env.poll(say=lambda *_: None)
    (env.bin / "status").write_text("COMPLETE"); env.clk.add(minutes=6); env.poll(say=lambda *_: None)   # charged 0.1 h
    env.launch(say=lambda *_: None)                                        # 0.1 + 0.5 fits
    env.clk.add(days=5)                                                    # new quota week: the old record no longer applies
    env.launch(say=lambda *_: None)
    assert len(pushes(env)) == 3


def test_run_loop_launches_in_order_within_limits_and_stops(env):
    env.add("run_a", env.nb, 60); env.add("run_b", env.nb, 60); env.add("run_c", env.nb, 60)
    (env.bin / "status").write_text("RUNNING")
    ticks = []
    def sleep(_):
        ticks.append(1)
        env.clk.add(minutes=30)
        if len(ticks) == 2:
            (env.bin / "status").write_text("COMPLETE")
    why = env.run_loop(every=1, say=lambda *_: None, sleep=sleep)
    assert [c.split()[3].rsplit("/", 1)[1] for c in pushes(env)] == ["run_a", "run_b", "run_c"]
    assert why == "nothing queued" and all(r["state"] == "complete" for r in env.load()["runs"])


def test_run_loop_stops_launching_on_halt(env):
    env.add("run_a", env.nb, 60); env.add("run_b", env.nb, 60); env.add("run_c", env.nb, 60)
    (env.bin / "status").write_text("ERROR")
    why = env.run_loop(every=1, say=lambda *_: None, sleep=lambda _: env.clk.add(minutes=5))
    assert why == "HALT is set" and len(pushes(env)) == 2


def test_run_loop_waits_for_start_time_before_launching(env):
    env.add("run_a", env.nb, 60)
    (env.bin / "status").write_text("COMPLETE")
    t0 = env.clock() + __import__("datetime").timedelta(hours=2)
    seen = []
    def sleep(sec):
        seen.append((env.clock() < t0, len(pushes(env))))
        env.clk.add(seconds=sec)
    why = env.run_loop(every=600, say=lambda *_: None, sleep=sleep, start_at=t0.strftime("%Y-%m-%dT%H:%M:%SZ"))
    waits = [s for s in seen if s[0]]
    assert len(waits) >= 12 and all(n == 0 for _, n in waits)       # nothing pushed while waiting
    assert len(pushes(env)) == 1 and why == "nothing queued"


def test_cap_override_applies_to_its_week_only(env):
    env.clk.t = dt.datetime(2026, 11, 23, 3, 0, tzinfo=dt.timezone.utc)   # week of 21 Nov: 29.5 h (owner, 23 Nov)
    env.external(26.0, "by hand")
    env.add("run_a", env.nb, 180); env.add("run_b", env.nb, 60)
    env.launch(say=lambda *_: None)                                        # 26 + 3 = 29 <= 29.5
    with pytest.raises(K.Refused, match="cap is 29.5"):                    # 26 + 3 in flight + 1 > 29.5
        env.launch(say=lambda *_: None)
    env.clk.t = dt.datetime(2026, 11, 28, 1, 0, tzinfo=dt.timezone.utc)   # next week: back to 26
    assert env.ledger(env.load())["cap"] == 26.0


def test_kernel_sources_are_passed_and_checked(env):
    with pytest.raises(K.Refused, match="kernel source"):
        env.add("run_x", env.nb, 10, kernel_sources=["not a slug"])
    env.add("run_a", env.nb, 10, kernel_sources=["tester/ft-pair-a0-b0"])
    env.launch(say=lambda *_: None)
    meta = json.loads((env.kdir / "push" / "run_a" / "kernel-metadata.json").read_text())
    assert meta["kernel_sources"] == ["tester/ft-pair-a0-b0"]
