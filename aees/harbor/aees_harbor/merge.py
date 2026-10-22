"""Merge ledgers from several machines into one, refusing anything inconsistent.

Designed for this setup: every machine starts from the same seed ledger and runs a disjoint
set of tasks (scenarios). Then the union of all rows is consistent by construction: rows from
the seed are identical everywhere, and new rows never share a key. The merge checks that:
a row whose key already exists must be identical in content, otherwise nothing is written.

    python -m aees_harbor.merge --into state/ledger.sqlite i1.sqlite i2.sqlite \\
        --rewrite /home/ubuntu/jobs=/Users/me/code/jobs

--rewrite maps trial-dir path prefixes (e.g. EC2 paths to where the jobs were rsynced).
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

from aees_harbor.ledger import Ledger

# table -> (key columns, columns that must match when the key already exists)
TABLES = {
    "scenarios": (["scenario_key"], ["fields_json"]),
    "runs": (["scenario_key", "run_index"], ["outcome", "reward", "trial_dir", "metrics_json"]),
    "trials": (["scenario_key", "run_index", "attempt"], ["trial_dir", "facts_json"]),
    "infra_attempts": (["scenario_key", "run_index", "try"], ["error_class", "trial_dir"]),
}  # leases are transient and never merged


def _rewrite(path, rules: list[tuple[str, str]]):
    if path is None:
        return None
    for old, new in rules:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def merge(into: Path, sources: list[Path], rewrite: list[tuple[str, str]] | None = None) -> dict:
    """Merge sources into `into` in one transaction. Raises ValueError (and writes nothing) on conflict."""
    rules = rewrite or []
    Ledger(into).close()  # create the schema if `into` is new
    db = sqlite3.connect(str(into), isolation_level=None)
    report = {t: {"inserted": 0, "identical": 0} for t in TABLES}
    conflicts: list[str] = []
    db.execute("BEGIN IMMEDIATE")
    try:
        for src in sources:
            s = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
            for table, (key_cols, check_cols) in TABLES.items():
                cols = [r[1] for r in s.execute(f"PRAGMA table_info({table})")]
                for row in s.execute(f"SELECT {', '.join(cols)} FROM {table}"):
                    rec = dict(zip(cols, row))
                    if "trial_dir" in rec:
                        rec["trial_dir"] = _rewrite(rec["trial_dir"], rules)
                    where = " AND ".join(f"{c} = ?" for c in key_cols)
                    existing = db.execute(
                        f"SELECT {', '.join(check_cols)} FROM {table} WHERE {where}",
                        [rec[c] for c in key_cols]).fetchone()
                    if existing is None and table == "trials":  # trial_dir is also unique
                        clash = db.execute("SELECT scenario_key, run_index, attempt FROM trials "
                                           "WHERE trial_dir = ?", (rec["trial_dir"],)).fetchone()
                        if clash:
                            conflicts.append(f"{src.name}: trials {rec['trial_dir']} already logged as {clash}")
                            continue
                    if existing is None:
                        db.execute(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                                   [rec[c] for c in cols])
                        report[table]["inserted"] += 1
                    elif list(existing) == [rec[c] for c in check_cols]:
                        report[table]["identical"] += 1
                        if table == "scenarios":  # keep the larger target if machines differed
                            db.execute("UPDATE scenarios SET target_runs = MAX(target_runs, ?) "
                                       "WHERE scenario_key = ?", (rec["target_runs"], rec["scenario_key"]))
                    else:
                        key = {c: rec[c] for c in key_cols}
                        conflicts.append(f"{src.name}: {table} {key} differs from an existing row")
            s.close()
        if conflicts:
            raise ValueError(f"{len(conflicts)} conflicting rows; nothing merged. First: {conflicts[:5]}")
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    finally:
        db.close()
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sources", nargs="+", type=Path)
    p.add_argument("--into", type=Path, required=True)
    p.add_argument("--rewrite", action="append", default=[], metavar="OLD=NEW",
                   help="rewrite trial-dir prefix OLD to NEW (repeatable)")
    a = p.parse_args()
    rules = [tuple(r.split("=", 1)) for r in a.rewrite]
    try:
        print(merge(a.into, a.sources, rules))
    except ValueError as e:
        sys.exit(f"MERGE REFUSED: {e}")


if __name__ == "__main__":
    main()
