"""Print the prebuilt Docker image of each selected task (one per line), for pre-pulling.

    harbor datasets download terminal-bench/terminal-bench-2-1 -o ~/tb21
    python scripts/task_images.py ~/tb21 state/tb21_selected_tasks.txt | xargs -n1 -P8 docker pull
"""
import sys
import tomllib
from pathlib import Path

root, tasks_file = Path(sys.argv[1]).expanduser(), Path(sys.argv[2])
wanted = [t.strip() for t in tasks_file.read_text().split() if t.strip()]
found = {p.parent.name: p for p in root.rglob("task.toml")}
missing = [t for t in wanted if t not in found]
if missing:
    sys.exit(f"tasks not found under {root}: {missing}")
for t in wanted:
    env = tomllib.loads(found[t].read_text()).get("environment", {})
    if env.get("docker_image"):
        print(env["docker_image"])
    else:
        print(f"# {t}: no prebuilt image (built from Dockerfile at run time)", file=sys.stderr)
