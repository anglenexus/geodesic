"""scripts/budget_guard.py: spend accounting from cloud listings, and the hard-cap kill against fake CLIs."""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("budget_guard", ROOT / "scripts" / "budget_guard.py")
bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bg)


def test_spend_counts_live_and_ended_instances():
    ledger = {}
    bg.update(ledger, "nebius", {"h200": {"rate": 5.4, "start": 0.0, "cloud": "nebius"}}, now=3600)
    bg.update(ledger, "ec2", {"b1": {"rate": 1.85, "start": 1800.0, "cloud": "ec2"}}, now=3600)
    assert bg.spend(ledger, 0) == pytest.approx(5.4 + 1.85 / 2)
    bg.update(ledger, "ec2", {}, now=7200)                       # box gone: closed at its last sighting (3600)
    bg.update(ledger, "nebius", {"h200": {"rate": 5.4, "start": 0.0, "cloud": "nebius"}}, now=7200)
    assert ledger["b1"]["end"] == 3600 and bg.spend(ledger, 0) == pytest.approx(2 * 5.4 + 1.85 / 2)
    assert bg.spend(ledger, 0.05) == pytest.approx(1.05 * (2 * 5.4 + 1.85 / 2))


def test_failed_listing_keeps_counting_and_a_reappearing_id_reopens():
    ledger = {}
    bg.update(ledger, "nebius", {"n": {"rate": 4.5, "start": 0.0, "cloud": "nebius"}}, now=100)
    bg.update(ledger, "nebius", None, now=3600)                   # API down: assume still running
    assert ledger["n"]["end"] is None and bg.spend(ledger, 0) == pytest.approx(4.5)
    bg.update(ledger, "nebius", {}, now=4000)
    assert ledger["n"]["end"] == 3600


def test_rates_and_creation_times_come_from_the_listings(tmp_path, monkeypatch):
    fake = fake_clis(tmp_path, nebius=[instance("ci-1", "gpu-h200-sxm", "1gpu-16vcpu-200gb", hours_ago=2)],
                     ec2=[["i-1", "m7a.8xlarge", hours_ago(1)]])
    monkeypatch.setenv("PATH", f"{fake}:{os.environ['PATH']}")
    n, e = bg.nebius_live(), bg.ec2_live()
    assert n["ci-1"]["rate"] == 5.4 and abs(time.time() - n["ci-1"]["start"] - 7200) < 60
    assert e["i-1"]["rate"] == 1.85 and abs(time.time() - e["i-1"]["start"] - 3600) < 60


def test_hard_cap_deletes_everything_and_exits(tmp_path):
    fake = fake_clis(tmp_path, nebius=[instance("ci-1", "gpu-h200-sxm", "1gpu-16vcpu-200gb", hours_ago=30)],
                     ec2=[["i-1", "m7a.8xlarge", hours_ago(10)], ["i-2", "m7a.8xlarge", hours_ago(10)]])
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}
    state = tmp_path / "state"
    r = subprocess.run([sys.executable, ROOT / "scripts" / "budget_guard.py", "--hard", "180", "--soft", "165",
                        "--soft-cmd", f"touch {tmp_path / 'soft_ran'}", "--poll", "0.1", "--state", state],
                       env=env, capture_output=True, text=True, timeout=60)
    assert "BUDGET_KILLED" in r.stdout, r.stdout + r.stderr          # 30 h × 5.4 + 2 × 10 h × 1.85 = 199 > 180
    assert json.loads((tmp_path / "nebius.json").read_text()) == []
    assert json.loads((tmp_path / "ec2.json").read_text()) == []
    assert (state / "budget_killed").exists() and (state / "budget_soft_done").exists()
    time.sleep(0.5)
    assert (tmp_path / "soft_ran").exists()


def test_under_the_cap_nothing_is_touched(tmp_path):
    fake = fake_clis(tmp_path, nebius=[instance("ci-1", "gpu-h100-sxm", "1gpu-16vcpu-200gb", hours_ago=1)], ec2=[])
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}
    r = subprocess.run([sys.executable, ROOT / "scripts" / "budget_guard.py", "--once", "--state", tmp_path / "s"],
                       env=env, capture_output=True, text=True, timeout=60)
    assert "spend $4.7" in r.stdout and "1 live" in r.stdout              # 4.50 × 1.05
    assert len(json.loads((tmp_path / "nebius.json").read_text())) == 1
    assert not (tmp_path / "s" / "budget_killed").exists()


# --- fake nebius / aws CLIs backed by JSON files ---

def hours_ago(h: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - h * 3600))


def instance(iid, platform, preset, hours_ago):
    return {"metadata": {"id": iid, "created_at": globals()["hours_ago"](hours_ago)},
            "spec": {"resources": {"platform": platform, "preset": preset}}}


def fake_clis(tmp: Path, nebius: list, ec2: list) -> Path:
    (tmp / "nebius.json").write_text(json.dumps(nebius))
    (tmp / "ec2.json").write_text(json.dumps(ec2))
    bin_dir = tmp / "bin"
    bin_dir.mkdir()
    (bin_dir / "nebius").write_text(f"""#!{sys.executable}
import json, sys
f = "{tmp / 'nebius.json'}"; items = json.load(open(f)); a = sys.argv[1:]
if "list" in a:
    proj = a[a.index("--parent-id") + 1]
    print(json.dumps({{"items": items if proj.startswith("project-u00") else []}}))
elif "delete" in a:
    i = a[a.index("--id") + 1]; json.dump([x for x in items if x["metadata"]["id"] != i], open(f, "w"))
""")
    (bin_dir / "aws").write_text(f"""#!{sys.executable}
import json, sys
f = "{tmp / 'ec2.json'}"; items = json.load(open(f)); a = sys.argv[1:]
if "describe-instances" in a:
    print(json.dumps(items))
elif "terminate-instances" in a:
    ids = a[a.index("--instance-ids") + 1:]; json.dump([x for x in items if x[0] not in ids], open(f, "w"))
""")
    for f in bin_dir.iterdir():
        f.chmod(0o755)
    return bin_dir
