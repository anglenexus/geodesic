"""Copy real Harbor trial dirs into small, labelled test fixtures.

Run once from aees/harbor:  python tests/fixtures/make_fixtures.py ~/code/jobs
Large text logs are trimmed to their tail; recordings and trajectories are not copied.
"""
import glob
import shutil
import sys
from pathlib import Path

JOBS = Path(sys.argv[1]).expanduser()
OUT = Path(__file__).parent / "trials"
TAIL_LINES = 300

# label -> trial dir (relative to JOBS; a trailing * matches Harbor's random trial suffix)
SOURCES = {
    "platform_manifest_arm64": "2026-10-01__18-45-40/astropy__astropy-12907__r4u55Qp",
    "infra_registry_429": "2026-10-01__18-45-40/astropy__astropy-13398__8Pq4pGn",
    "infra_verifier_network_drop": "2026-10-02__15-31-20/psf__requests-2317__jqiF5G7",
    "pass_oracle_swebench": "2026-10-02__15-31-20/psf__requests-1142__tqtiz4K",
    "fail_qwen_swebench": "2026-10-02__20-18-35/psf__requests-1142__E6M5w46",
    "fatal_no_credits": "luna_smoke/psf__requests-1142__WsMsT4f",
    "fail_luna_swebench": "luna_smoke2/psf__requests-1142__BbYkyqr",
    "pass_luna_swebench": "luna_smoke2/psf__requests-2317__SjFGajX",
    "pass_oracle_tb21": "tb21_oracle_phase0/extract-elf__*",
    "fail_oracle_tb21_env_drift": "tb21_oracle_phase0/mcmc-sampling-stan__*",
    "fail_oracle_tb21_stochastic": "tb21_oracle_phase0/protein-assembly__*",
    "platform_rosetta_syscall": "tb21_oracle_phase0/qemu-startup__*",
    "platform_illegal_instruction": "tb21_oracle_phase0/tune-mjcf__*",
}
COPY = ["result.json", "exception.txt", "trial.log", "verifier/reward.txt", "verifier/ctrf.json",
        "verifier/report.json", "verifier/test-stdout.txt", "agent/oracle.txt", "agent/terminus_2.pane"]
TRIM = {"trial.log", "test-stdout.txt", "oracle.txt", "terminus_2.pane"}


def main() -> None:
    for label, rel in SOURCES.items():
        [src] = glob.glob(str(JOBS / rel))
        dst = OUT / label
        shutil.rmtree(dst, ignore_errors=True)
        for name in COPY:
            s = Path(src) / name
            if not s.exists():
                continue
            d = dst / name
            d.parent.mkdir(parents=True, exist_ok=True)
            if s.name in TRIM:
                lines = s.read_text(errors="replace").splitlines(keepends=True)
                d.write_text("".join(lines[-TAIL_LINES:]))
            else:
                shutil.copyfile(s, d)
        print(f"{label:32s} <- {Path(src).relative_to(JOBS)}")


if __name__ == "__main__":
    main()
