"""grpo/command_timeout.patch: Terminus-2's 120 s per-command limit (align_tmax_plan.md, item 5a).

Both tests need a Harbor checkout and skip without one:
  HARBOR_GIT=<clone of harbor-framework/harbor>    the patch applies at the pinned commit, after harbor_parser.patch
  HARBOR_SRC=<that checkout, both patches applied, `uv sync` done>   the limit's behavior, with a fake terminal
"""
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATCHES = [ROOT / "grpo/harbor_parser.patch", ROOT / "grpo/command_timeout.patch"]


def pinned_harbor_commit() -> str:
    pins = (ROOT / "configs/pins.sh").read_text()
    return pins.split("HARBOR_COMMIT=${HARBOR_COMMIT:-", 1)[1].split("}", 1)[0]


@pytest.mark.skipif(not os.environ.get("HARBOR_GIT"), reason="set HARBOR_GIT to a harbor clone")
def test_patch_applies_at_the_pinned_commit(tmp_path):
    work = tmp_path / "harbor"
    subprocess.run(["git", "clone", "-q", "--no-checkout", os.environ["HARBOR_GIT"], str(work)], check=True)
    subprocess.run(["git", "-C", str(work), "checkout", "-q", "--detach", pinned_harbor_commit()], check=True)
    for patch in PATCHES:                                  # the order setup.sh applies them in
        r = subprocess.run(["git", "-C", str(work), "apply", "--check", str(patch)], capture_output=True, text=True)
        assert r.returncode == 0, f"{patch.name}: {r.stderr}"
        subprocess.run(["git", "-C", str(work), "apply", str(patch)], check=True)


BEHAVIOR = textwrap.dedent('''
    import asyncio, logging
    from harbor.agents.terminus_2.terminus_2 import Command, Terminus2

    class FakeEnv:
        def __init__(self, session):
            self.session, self.execs = session, []
        async def exec(self, command, user=None):
            self.execs.append(command)
            if command.startswith("kill -KILL"):
                self.session.job = 100

    class FakeSession:
        """A terminal whose foreground group is the shell's (100) at the prompt, else a job's (None: unreadable)."""
        def __init__(self, job=100, ignores_sigint=False, broken=False):
            self.job, self.ignores_sigint, self.broken = job, ignores_sigint, broken
            self.sent, self.checks, self._user = [], 0, None
            self.environment = FakeEnv(self)
        def runs_command(self, keys):
            return keys.endswith("\\n")
        async def foreground_job(self):
            self.checks += 1
            if self.broken:
                raise RuntimeError("docker exec failed")
            return self.job
        async def send_keys(self, keys, block=False, min_timeout_sec=0.0):
            self.sent.append(keys)
            if keys == "C-c" and not self.ignores_sigint:
                self.job = 100
            elif keys.startswith(("sleep", "bash ")):
                self.job = 4242
        async def get_incremental_output(self):
            return "screen"

    def agent(limit):
        t = Terminus2.__new__(Terminus2)
        t._command_timeout_sec, t._command_seq, t._command_notes = limit, 0, []
        t._command_watchdogs, t._command_exec_times, t._shell_job = set(), [], None
        t._limit_output_length = lambda s: s
        t.logger = logging.getLogger("test")
        return t

    async def run(t, s, keys, wait):
        await t._execute_commands([Command(keys, 0.05)], s)
        await asyncio.sleep(wait)

    async def main():
        out = {}
        # 1. a command still running at the limit gets one Ctrl-C; the next output says so
        t, s = agent(0.3), FakeSession()
        await run(t, s, "sleep 999\\n", 2.6)
        _, text = await t._execute_commands([Command("ls\\n", 0.05)], s)
        out["long"] = (s.sent.count("C-c"), len(s.environment.execs), text.splitlines()[0])
        # 2. a script run by bash is a job too (a name check would see "bash" and miss it)
        t, s = agent(0.3), FakeSession()
        await run(t, s, "bash build.sh\\n", 2.6)
        out["script"] = s.sent.count("C-c")
        # 3. a command that finishes in time is left alone
        t, s = agent(0.3), FakeSession()
        await run(t, s, "ls\\n", 0.5)
        out["short"] = s.sent.count("C-c")
        # 4. keys typed into a running program (here an editor, after a first command) start no timer
        t, s = agent(0.3), FakeSession()
        await t._execute_commands([Command("ls\\n", 0.05)], s)
        s.job = 777
        n = len(t._command_watchdogs)
        await t._execute_commands([Command(":wq\\n", 0.05)], s)
        out["interactive"] = (s.sent.count("C-c"), len(t._command_watchdogs) - n)
        # 5. a command that finished before a newer one started is not interrupted later
        t, s = agent(0.3), FakeSession()
        await t._execute_commands([Command("sleep 999\\n", 0.05)], s)
        s.job = 100
        await t._execute_commands([Command("echo hi\\n", 0.05)], s)
        s.job = 999
        await asyncio.sleep(0.27)
        out["superseded"] = s.sent.count("C-c")
        # 6. a process that ignores Ctrl-C is killed by process group
        t, s = agent(0.3), FakeSession(ignores_sigint=True)
        await run(t, s, "sleep 999\\n", 2.8)
        _, text = await t._execute_commands([Command("ls\\n", 0.05)], s)
        out["stubborn"] = (s.environment.execs, "killed" in text.splitlines()[0])
        # 7. an unreadable terminal: no timer, no stray Ctrl-C, the trial goes on
        t, s = agent(0.3), FakeSession(broken=True)
        await run(t, s, "sleep 999\\n", 0.5)
        out["broken"] = (s.sent.count("C-c"), len(t._command_watchdogs))
        # 8. no limit: no checks, no timers
        t, s = agent(None), FakeSession()
        await run(t, s, "sleep 999\\n", 0.1)
        out["off"] = (s.checks, len(t._command_watchdogs))
        print(repr(out))

    asyncio.run(main())
''')


@pytest.mark.skipif(not os.environ.get("HARBOR_SRC"), reason="set HARBOR_SRC to a patched harbor checkout")
def test_command_limit_behavior():
    uv = shutil.which("uv")
    assert uv, "uv not found"
    r = subprocess.run([uv, "run", "--directory", os.environ["HARBOR_SRC"], "python", "-c", BEHAVIOR],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    out = eval(r.stdout.strip().splitlines()[-1])
    assert out["long"][:2] == (1, 0)                       # one Ctrl-C, no kill needed
    assert out["long"][2].startswith("Command timed out after 0.3 seconds and was interrupted with Ctrl-C: sleep 999")
    assert out["script"] == 1
    assert out["short"] == 0
    assert out["interactive"] == (0, 0)
    assert out["superseded"] == 0
    assert out["stubborn"] == (["kill -KILL -- -4242"], True)
    assert out["broken"] == (0, 0)
    assert out["off"] == (0, 0)


WIRING = textwrap.dedent('''
    import os
    from agent_server.models import RunRequest
    from agent_server.trial_runner import _agent_connection_config
    req = RunRequest(base_url="http://x/v1", model="openai/model", agent_name="terminus-2")
    off, _ = _agent_connection_config(req)
    os.environ["HARBOR_TERMINUS_2_COMMAND_TIMEOUT_SEC"] = "120"
    os.environ["HARBOR_TERMINUS_2_STOP_AT_CONTEXT"] = "1"
    on, _ = _agent_connection_config(req)
    print(repr(({k: off.get(k) for k in ("command_timeout_sec", "stop_at_context_limit")},
                {k: on.get(k) for k in ("command_timeout_sec", "stop_at_context_limit")})))
''')


@pytest.mark.skipif(not os.environ.get("HARBOR_SRC"), reason="set HARBOR_SRC to a patched harbor checkout")
def test_agent_server_passes_the_tmax_options_only_when_set():
    env = {k: v for k, v in os.environ.items() if not k.startswith("HARBOR_TERMINUS_2_")}
    r = subprocess.run([shutil.which("uv"), "run", "--directory", os.environ["HARBOR_SRC"], "python", "-c", WIRING],
                       capture_output=True, text=True, timeout=120, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    off, on = eval(r.stdout.strip().splitlines()[-1])
    assert off == {"command_timeout_sec": None, "stop_at_context_limit": None}
    assert on == {"command_timeout_sec": 120.0, "stop_at_context_limit": True}
