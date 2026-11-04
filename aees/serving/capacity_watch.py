#!/usr/bin/env python3
"""
capacity_watch.py - from the laptop: watch for a regular (on-demand, not preemptible) 8x B300 node and take it.

    python capacity_watch.py                  # every 15 min until a node is secured or 24 h pass
    python capacity_watch.py --once           # one round of checks
    python capacity_watch.py --no-launch      # Nebius: report capacity but delete the VM instead of running

Each round:
  - Nebius (aees CLI profile): try to create a regular 8x B300 VM in us-north1, uk-south1 and eu-west2, the regions
    where the tenant has B300 quota. Nebius has no availability API, so a create attempt is the test; a failed
    attempt leaves a STOPPED instance with no charge, which is deleted. On success the VM is kept and
    nebius_kimi_k3.sh has already started the run (kill switch, S3 sync started here), and the watch stops.
  - Prime Intellect (if PRIME_API_KEY is set, or in ~/.prime_api_key): list 8x B300 offers that are in stock
    and not spot. Reported only (not booked).
  - Vast.ai (public search API, no key): rentable 8x B300 offers of type on-demand (not interruptible). Reported only.
  - Shadeform (if SHADEFORM_API_KEY is set, or in ~/.shadeform_api_key): aggregator of Lambda, Nebius, Crusoe,
    Hyperstack and others; available 8x B300 instance types. Reported only.
Every check is appended to .nebius/capacity_log.jsonl; lines starting with ">>" are the ones worth reading.
"""
import argparse
import datetime
import json
import os
import pathlib
import re
import subprocess
import time
import urllib.parse
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
STATE = HERE / ".nebius"
LOG = STATE / "capacity_log.jsonl"
NEBIUS = [os.environ.get("NEBIUS_BIN", str(pathlib.Path.home() / ".nebius/bin/nebius")), "--profile", "aees"]
REGIONS = [  # (region, project, subnet): the tenant's B300 quota regions
    ("us-north1", "project-u02n20nqig001ntg5jdh7b", "vpcsubnet-u02n7ccew5fnhz00nw"),
    ("uk-south1", "project-e03m88qnlc00z22k2eg00f", "vpcsubnet-e03n1xpx7an3yb9ms5"),
    ("eu-west2", "project-e04y3zw0bc00fjscrh8w3s", "vpcsubnet-e04hvt3a5pk5vrdawr"),
]


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def record(**kw):
    kw = {"time": now(), **kw}
    STATE.mkdir(exist_ok=True)
    with open(LOG, "a") as f:
        f.write(json.dumps(kw) + "\n")
    return kw


def say(msg):
    print(f"[{now()}] {msg}", flush=True)


def nebius_try(region, project, subnet, launch):
    """One create attempt; returns True if a VM is running."""
    (STATE / "vm.env").unlink(missing_ok=True)
    env = dict(os.environ, PROJECT=project, SUBNET=subnet, PREEMPTIBLE="0", NAME=f"kimi-k3-b300-{region}")
    t = time.time()
    p = subprocess.run(["bash", str(HERE / "nebius_kimi_k3.sh"), "up"], env=env, capture_output=True, text=True)
    out = p.stdout + p.stderr
    secs = round(time.time() - t)
    if p.returncode == 0 and "Setup started" in out:
        record(provider="nebius", region=region, result="SECURED", seconds=secs)
        say(f">> NEBIUS {region}: regular 8x B300 SECURED after {secs} s")
        if not launch:
            subprocess.run(["bash", str(HERE / "nebius_kimi_k3.sh"), "down"], env=env)
            say(">> --no-launch: deleted it again")
        else:
            with open(STATE / "vm.env", "a") as f:
                f.write(f"REGION={region}\nPROJECT={project}\nSUBNET={subnet}\n")
        return True
    reason = "not enough resources" if "NotEnoughResources" in out else (out.strip().splitlines() or ["?"])[-1][:200]
    record(provider="nebius", region=region, result="unavailable", reason=reason, seconds=secs)
    say(f"nebius {region}: {reason} ({secs} s)")
    for vm in set(re.findall(r"computeinstance-[a-z0-9]+", out)):  # the STOPPED leftover of a failed create
        subprocess.run(NEBIUS + ["compute", "instance", "delete", "--id", vm], stdin=subprocess.DEVNULL,
                       capture_output=True)
    (STATE / "vm.env").unlink(missing_ok=True)
    return False


def prime_key():
    if os.environ.get("PRIME_API_KEY"):
        return os.environ["PRIME_API_KEY"]
    f = pathlib.Path.home() / ".prime_api_key"
    if f.exists():
        m = re.search(r"PRIME_API_KEY=(\S+)", f.read_text())
        return m.group(1) if m else None
    return None


def prime_check(key):
    offers = []
    for gpu_type in ("B300_262GB", "B300_288GB", "B300"):
        url = f"https://api.primeintellect.ai/api/v1/availability/gpus?gpu_type={gpu_type}&gpu_count=8&page_size=100"
        try:
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
            with urllib.request.urlopen(req, timeout=30) as r:
                items = json.load(r).get("items", [])
        except Exception as e:
            if "422" not in str(e) and "400" not in str(e):  # unknown gpu_type names just don't match
                record(provider="prime", gpu_type=gpu_type, result="error", reason=str(e)[:200])
            continue
        for it in items:
            if it.get("isSpot") or it.get("stockStatus") == "Unavailable" or it.get("gpuCount") != 8:
                continue
            pr = it.get("prices") or {}
            offers.append(dict(provider=it.get("provider"), region=it.get("region"), dc=it.get("dataCenter"),
                               stock=it.get("stockStatus"), security=it.get("security"),
                               price=pr.get("onDemand") or pr.get("communityPrice"), cloud_id=it.get("cloudId")))
        if items:
            break
    record(provider="prime", result="offers" if offers else "none", offers=offers)
    if offers:
        for o in offers:
            say(f">> PRIME: regular 8x B300 at {o['provider']} {o['region']} ({o['stock']}, {o['security']}): "
                f"${o['price']}/GPU-h, cloud_id {o['cloud_id']}")
    else:
        say("prime: no in-stock, non-spot 8x B300 offers")


def keyfile(var, path):
    if os.environ.get(var):
        return os.environ[var]
    f = pathlib.Path.home() / path
    m = re.search(var + r"=(\S+)", f.read_text()) if f.exists() else None
    return m.group(1) if m else None


def report(provider, offers, none_text):
    record(provider=provider, result="offers" if offers else "none", offers=offers)
    for o in offers:
        say(f">> {provider.upper()}: regular 8x B300 at {o['provider']} {o['region']}: ${o['price']}/h "
            f"({o.get('note', '')})")
    if not offers:
        say(f"{provider}: {none_text}")


def vast_check():
    q = json.dumps({"num_gpus": {"eq": 8}, "rentable": {"eq": True}, "type": "on-demand"})
    try:
        with urllib.request.urlopen("https://console.vast.ai/api/v0/bundles/?q=" + urllib.parse.quote(q),
                                    timeout=30) as r:
            items = json.load(r).get("offers", [])
    except Exception as e:
        record(provider="vast", result="error", reason=str(e)[:200])
        return say(f"vast: error {str(e)[:120]}")
    offers = [dict(provider="vast.ai", region=o.get("geolocation"), price=round(o.get("dph_total") or 0, 2),
                   note=f"{o.get('gpu_name')}, reliability {o.get('reliability2', 0):.3f}, id {o.get('id')}")
              for o in items if "B300" in str(o.get("gpu_name", "")).upper()]
    report("vast", offers, "no rentable on-demand 8x B300")


def shadeform_check(key):
    url = "https://api.shadeform.ai/v1/instances/types?num_gpus=8&available=true"
    try:
        req = urllib.request.Request(url, headers={"X-API-KEY": key})
        with urllib.request.urlopen(req, timeout=30) as r:
            types = json.load(r).get("instance_types", [])
    except Exception as e:
        record(provider="shadeform", result="error", reason=str(e)[:200])
        return say(f"shadeform: error {str(e)[:120]}")
    offers = []
    for t in types:
        if "B300" not in str(t.get("gpu_type", "")).upper():
            continue
        regions = [a.get("region") for a in t.get("availability", []) if a.get("available")]
        if regions:
            offers.append(dict(provider=t.get("cloud"), region=",".join(regions),
                               price=round((t.get("hourly_price") or 0) / 100, 2), note=t.get("shade_instance_type")))
    report("shadeform", offers, "no available 8x B300")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interval-min", type=float, default=15)
    ap.add_argument("--hours", type=float, default=24)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--no-launch", action="store_true")
    a = ap.parse_args()
    end = time.time() + a.hours * 3600
    rnd = 0
    while True:
        rnd += 1
        say(f"round {rnd}")
        key = prime_key()
        if key:
            prime_check(key)
        vast_check()
        sf = keyfile("SHADEFORM_API_KEY", ".shadeform_api_key")
        if sf:
            shadeform_check(sf)
        for region, project, subnet in REGIONS:
            if nebius_try(region, project, subnet, launch=not a.no_launch):
                if not a.no_launch:
                    say(">> run started on the VM; S3 sync running: .nebius/sync.log. "
                        "Tunnel: bash nebius_kimi_k3.sh tunnel")
                    with open(STATE / "sync.log", "a") as f:
                        subprocess.Popen(["bash", str(HERE / "nebius_kimi_k3.sh"), "sync"], stdout=f,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                return
        if a.once or time.time() > end:
            say("watch finished without securing a node")
            return
        time.sleep(a.interval_min * 60)


if __name__ == "__main__":
    main()
