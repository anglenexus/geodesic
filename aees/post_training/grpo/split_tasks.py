"""Task lists -> the prompt files GRPO reads: one jsonl row per task, for training and validation.

Miles passes each row's metadata to the Harbor agent server, which runs terminus-2 on the task folder
HARBOR_TASKS_DIR/<instance_id>. Harbor builds the actual prompt from that folder's instruction.md, so
"prompt" only names the task. Row format (Miles examples/experimental/terminus-compaction/README.md):
    {"prompt": "mailman", "metadata": {"instance_id": "mailman", "agent_name": "terminus-2"}}
The test list is read only to check that the three sets don't overlap (D12); no test file is written.

    uv run python grpo/split_tasks.py --tasks-dir /data/tasks/terminal-bench-2-1 --out-dir data
"""
import argparse
import json
from pathlib import Path

STATE = Path(__file__).resolve().parents[2] / "harbor" / "state"     # aees/harbor/state: the task lists
LISTS = {"train": "tb21_selected_tasks.txt", "val": "tb21_val8_tasks.txt", "test": "tb21_test20_tasks.txt"}
AGENT = "terminus-2"


def read_list(path: Path) -> list[str]:
    """One task name per line; blank lines skipped."""
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    assert len(names) == len(set(names)), f"duplicate task in {path}"
    return names


def row(task: str) -> dict:
    return {"prompt": task, "metadata": {"instance_id": task, "agent_name": AGENT}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks-dir", type=Path, required=True, help="TB 2.1 task folders (= HARBOR_TASKS_DIR)")
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    ap.add_argument("--state", type=Path, default=STATE, help="folder holding the three task lists")
    args = ap.parse_args()
    sets = {name: read_list(args.state / file) for name, file in LISTS.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(sets[a]) & set(sets[b]), f"{a} and {b} overlap: {set(sets[a]) & set(sets[b])}"
    missing = [t for names in sets.values() for t in names if not (args.tasks_dir / t / "task.toml").exists()]
    assert not missing, f"task folders missing under {args.tasks_dir}: {missing}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["train", "val"]:
        out = args.out_dir / f"grpo_{name}.jsonl"
        out.write_text("".join(json.dumps(row(t)) + "\n" for t in sets[name]))
        print(f"{len(sets[name])} {name} tasks -> {out}")


if __name__ == "__main__":
    main()
