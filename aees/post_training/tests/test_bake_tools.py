"""scripts/bake_tools.py: tmux + asciinema baked into task images under the same names, with a fake docker."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import bake_tools  # noqa: E402


class FakeDocker:
    """has: images whose probe passes; users: Config.User per local image; broken: images whose build fails."""

    def __init__(self, has=(), users=None, broken=()):
        self.has, self.users, self.broken, self.calls = set(has), users or {}, set(broken), []

    def __call__(self, cmd, input=None, capture_output=True, text=True):
        self.calls.append((cmd, input))
        args = cmd[cmd.index("docker") + 1:]
        ok = lambda out="": subprocess.CompletedProcess(cmd, 0, out, "")
        bad = lambda err: subprocess.CompletedProcess(cmd, 1, "", err)
        if args[0] == "run":
            return ok("tmux 3.4") if args[-3] in self.has else bad("tmux: not found")
        if args[0] == "image":
            return ok(self.users[args[-1]] + "\n") if args[-1] in self.users else bad("No such image")
        if args[0] == "build":
            image = args[args.index("-t") + 1]
            return bad("E: Unable to locate package tmux") if image in self.broken else ok("sha256:abc")
        raise AssertionError(cmd)


def test_dockerfile_installs_as_root_checks_and_restores_the_user():
    d = bake_tools.dockerfile("alexgshaw/mailman:20251031", "app")
    lines = d.splitlines()
    assert lines[0] == "FROM alexgshaw/mailman:20251031" and lines[1] == "USER root"
    assert lines[2].startswith("RUN if command -v apt-get") and lines[2].endswith("&& tmux -V && asciinema --version")
    for pm in ("apt-get install -y --no-install-recommends tmux asciinema", "apk add --no-cache tmux asciinema",
               "dnf install -y tmux asciinema", "yum install -y tmux asciinema", "exit 3"):
        assert pm in lines[2]
    assert lines[-1] == "USER app"
    assert bake_tools.dockerfile("x:1", "").splitlines()[-1].startswith("LABEL aees.tools")   # root stays the default


def test_bake_rebuilds_under_the_same_name_from_stdin():
    fake = FakeDocker(users={"a/x:1": ""})
    assert bake_tools.bake("a/x:1", ["sudo", "docker"], run=fake) == "baked"
    build = [(c, i) for c, i in fake.calls if "build" in c][0]
    assert build[0] == ["sudo", "docker", "build", "-q", "-t", "a/x:1", "-"] and build[1].startswith("FROM a/x:1\n")
    probe = fake.calls[0][0]
    assert probe[:2] == ["sudo", "docker"] and "--user" in probe and probe[-1] == bake_tools.CHECK


def test_bake_leaves_images_that_have_the_tools_or_fail():
    fake = FakeDocker(has={"a/has:1"}, users={"a/broken:1": "", "a/has:1": ""}, broken={"a/broken:1"})
    assert bake_tools.bake("a/has:1", ["docker"], run=fake) == "has tools"
    assert not [c for c, _ in fake.calls if "build" in c]
    assert bake_tools.bake("a/broken:1", ["docker"], run=fake).startswith("FAILED: build (E: Unable to locate")
    assert bake_tools.bake("a/missing:1", ["docker"], run=fake).startswith("FAILED: not found locally")


def test_main_prints_each_image_and_a_summary(monkeypatch, capsys):
    fake = FakeDocker(has={"h:1"}, users={"b:1": "", "f:1": "", "h:1": ""}, broken={"f:1"})
    monkeypatch.setattr(bake_tools.subprocess, "run", fake)
    bake_tools.main(["b:1", "h:1", "f:1", "--docker", "sudo docker", "--jobs", "2"])
    out = capsys.readouterr().out.splitlines()
    assert out[:3] == ["b:1: baked", "h:1: has tools", out[2]] and out[2].startswith("f:1: FAILED")
    assert out[3] == "bake_tools: 1 baked, 1 already had the tools, 1 failed (those install at trial time)"
