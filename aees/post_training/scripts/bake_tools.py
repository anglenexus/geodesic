"""Bake tmux and asciinema into Harbor task images, under the same image names (integration_issues.md: agent setup
over 120 s). terminus-2 otherwise installs both into every trial's container from the internet (p50 25 s, and a
slow mirror ends the trial at its 120 s limit); it skips the install when `tmux -V` and `asciinema --version` work.
Trials run the local image (the compose file has no pull policy), so nothing else changes.

    python3 scripts/bake_tools.py [--docker "sudo docker"] IMAGE...

Each image: 'has tools' (left alone), 'baked' (rebuilt with one more layer and its original USER), or 'FAILED ...'
(left as it was: the trial falls back to the runtime install). Always exits 0; the summary line counts each.
"""
import argparse
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor

TOOLS = "tmux asciinema"
CHECK = "tmux -V && asciinema --version"
INSTALL = (
    "if command -v apt-get >/dev/null; then export DEBIAN_FRONTEND=noninteractive && apt-get update"
    f" && apt-get install -y --no-install-recommends {TOOLS} && rm -rf /var/lib/apt/lists/*;"
    f" elif command -v apk >/dev/null; then apk add --no-cache {TOOLS};"
    f" elif command -v dnf >/dev/null; then dnf install -y {TOOLS};"
    f" elif command -v yum >/dev/null; then yum install -y {TOOLS};"
    " else echo 'no known package manager' >&2; exit 3; fi"
)


def dockerfile(image: str, user: str) -> str:
    """One layer on top of `image`, installed as root, then the image's own USER back (empty means root)."""
    return (f"FROM {image}\nUSER root\nRUN {INSTALL} && {CHECK}\nLABEL aees.tools=\"{TOOLS}\"\n"
            + (f"USER {user}\n" if user else ""))


def bake(image: str, docker: list[str], run=None) -> str:
    def sh(*args, stdin=None):
        return (run or subprocess.run)([*docker, *args], input=stdin, capture_output=True, text=True)

    probe = sh("run", "--rm", "--entrypoint", "sh", "--user", "root", image, "-c", CHECK)
    if probe.returncode == 0:
        return "has tools"
    user = sh("image", "inspect", "-f", "{{.Config.User}}", image)
    if user.returncode != 0:
        return f"FAILED: not found locally ({user.stderr.strip()[-200:]})"
    build = sh("build", "-q", "-t", image, "-", stdin=dockerfile(image, user.stdout.strip()))
    if build.returncode != 0:
        return f"FAILED: build ({build.stderr.strip()[-300:]})"
    return "baked"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("images", nargs="+")
    p.add_argument("--docker", default="docker", help='docker command, e.g. "sudo docker"')
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args(argv)
    docker = shlex.split(a.docker)
    with ThreadPoolExecutor(a.jobs) as pool:
        results = list(pool.map(lambda im: bake(im, docker), a.images))
    for image, r in zip(a.images, results):
        print(f"{image}: {r}", flush=True)
    counts = {k: sum(r.startswith(k) for r in results) for k in ("has tools", "baked", "FAILED")}
    print(f"bake_tools: {counts['baked']} baked, {counts['has tools']} already had the tools, "
          f"{counts['FAILED']} failed (those install at trial time)")


if __name__ == "__main__":
    main()
