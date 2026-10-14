"""Run ledger: the cache that guarantees each run index of a scenario executes once.

A scenario is one task / model / tool-use setting. Its run indices 1, 2, ... are
handed out by lease_next(). Every finished Harbor trial is logged unjudged (raw facts)
and its index is put on hold. A review then resolves a held index to pass/fail (final,
never run again) or releases it to be retried at the same index.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Scenario:
    """Everything that defines a task / model / tool-use setting."""

    dataset: str
    dataset_ref: str
    task: str
    task_checksum: str
    model: str
    agent: str
    agent_version: str
    tool_mode: str
    harbor_version: str
    agent_kwargs: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Stable id: sha256 over the canonical JSON of every field (first 16 hex chars)."""
        blob = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS scenarios (
  scenario_key TEXT PRIMARY KEY, fields_json TEXT NOT NULL, task TEXT NOT NULL,
  target_runs INTEGER NOT NULL, frozen_at REAL, created_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS runs (
  scenario_key TEXT NOT NULL, run_index INTEGER NOT NULL,
  outcome TEXT NOT NULL CHECK (outcome IN ('pass', 'fail', 'held')), reward REAL,
  trial_dir TEXT, metrics_json TEXT, finished_at REAL NOT NULL,
  PRIMARY KEY (scenario_key, run_index));
CREATE TABLE IF NOT EXISTS trials (
  scenario_key TEXT NOT NULL, run_index INTEGER NOT NULL, attempt INTEGER NOT NULL,
  trial_dir TEXT NOT NULL UNIQUE, facts_json TEXT NOT NULL, recorded_at REAL NOT NULL,
  PRIMARY KEY (scenario_key, run_index, attempt));
CREATE TABLE IF NOT EXISTS infra_attempts (
  scenario_key TEXT NOT NULL, run_index INTEGER NOT NULL, try INTEGER NOT NULL,
  error_class TEXT NOT NULL, error_message TEXT, trial_dir TEXT, at REAL NOT NULL,
  PRIMARY KEY (scenario_key, run_index, try));
CREATE TABLE IF NOT EXISTS leases (
  scenario_key TEXT NOT NULL, run_index INTEGER NOT NULL, job_name TEXT,
  leased_at REAL NOT NULL, PRIMARY KEY (scenario_key, run_index));
"""
_WHERE_INDEX = "WHERE scenario_key = ? AND run_index = ?"


class Ledger:
    """SQLite-backed ledger. Every multi-statement write is a single transaction."""

    def __init__(self, path: str | Path, max_infra_retries: int = 3):
        self.max_infra_retries = max_infra_retries
        self._db = sqlite3.connect(str(path), isolation_level=None)  # we issue BEGIN/COMMIT
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def _tx(self):
        """BEGIN IMMEDIATE takes the write lock up front, so two runners can't lease the same index."""
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        self._db.execute("COMMIT")

    def register(self, scenario: Scenario, target_runs: int) -> str:
        """Create the scenario if new. An existing one keeps its runs; its target can only grow."""
        fields = json.dumps(asdict(scenario), sort_keys=True)
        self._db.execute(
            "INSERT INTO scenarios VALUES (?, ?, ?, ?, NULL, ?) ON CONFLICT(scenario_key) "
            "DO UPDATE SET target_runs = MAX(target_runs, excluded.target_runs)",
            (scenario.key, fields, scenario.task, target_runs, time.time()),
        )
        return scenario.key

    def _indices(self, sql: str, key: str) -> set[int]:
        return {row[0] for row in self._db.execute(sql, (key,))}

    def _final(self, key: str) -> set[int]:
        """Indices with a reviewed pass/fail outcome. These are never run again."""
        return self._indices(
            "SELECT run_index FROM runs WHERE scenario_key = ? AND outcome != 'held'", key)

    def _held(self, key: str) -> set[int]:
        """Indices whose trial finished but hasn't been reviewed yet."""
        return self._indices(
            "SELECT run_index FROM runs WHERE scenario_key = ? AND outcome = 'held'", key)

    def _leased(self, key: str) -> set[int]:
        return self._indices("SELECT run_index FROM leases WHERE scenario_key = ?", key)

    def _abandoned(self, key: str) -> set[int]:
        """Indices released for retry max_infra_retries times without a final outcome. Skipped for good."""
        exhausted = self._indices(
            "SELECT run_index FROM infra_attempts WHERE scenario_key = ? "
            f"GROUP BY run_index HAVING COUNT(*) >= {int(self.max_infra_retries)}",
            key,
        )
        return exhausted - self._final(key) - self._held(key)

    def _scenario_row(self, key: str) -> tuple[int, float | None]:
        row = self._db.execute(
            "SELECT target_runs, frozen_at FROM scenarios WHERE scenario_key = ?", (key,)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown scenario {key!r}; call register() first")
        return row

    def lease_next(self, key: str, n: int, job_name: str = "") -> list[int]:
        """Reserve up to n run indices that are not final, held, leased or abandoned; smallest first.

        Held indices occupy a slot until reviewed, so scheduling never overshoots the target.
        """
        with self._tx():
            target, frozen_at = self._scenario_row(key)
            final, held, leased = self._final(key), self._held(key), self._leased(key)
            budget = 0 if frozen_at else max(0, min(n, target - len(final) - len(held) - len(leased)))
            blocked = final | held | leased | self._abandoned(key)
            taken, i = [], 1
            while len(taken) < budget:
                if i not in blocked:
                    taken.append(i)
                i += 1
            now = time.time()
            self._db.executemany(
                "INSERT INTO leases VALUES (?, ?, ?, ?)", [(key, i, job_name, now) for i in taken]
            )
        return taken

    def record_trial(self, key: str, run_index: int, trial_dir: str, facts: dict) -> int:
        """Log one finished Harbor trial's raw facts, unjudged, and put its index on hold.

        Idempotent per trial_dir. Returns this trial's attempt number for the index (1, 2, ...).
        """
        with self._tx():
            row = self._db.execute(
                "SELECT attempt FROM trials WHERE trial_dir = ?", (trial_dir,)).fetchone()
            if row:
                return row[0]
            attempt = self._db.execute(
                f"SELECT COUNT(*) FROM trials {_WHERE_INDEX}", (key, run_index)).fetchone()[0] + 1
            now = time.time()
            self._db.execute(
                "INSERT INTO trials VALUES (?, ?, ?, ?, ?, ?)",
                (key, run_index, attempt, trial_dir, json.dumps(facts, sort_keys=True), now),
            )
            self._db.execute(  # a final outcome, if any, is never downgraded to held
                "INSERT OR IGNORE INTO runs VALUES (?, ?, 'held', NULL, ?, NULL, ?)",
                (key, run_index, trial_dir, now),
            )
            self._db.execute(f"DELETE FROM leases {_WHERE_INDEX}", (key, run_index))
        return attempt

    def resolve(self, key: str, run_index: int, decision: str, reward: float | None = None,
                reason: str = "") -> bool:
        """Review a held index: 'pass' / 'fail' make it final; 'retry' releases it to run again.

        Returns False, and changes nothing, if the index isn't currently held.
        """
        if decision not in ("pass", "fail", "retry"):
            raise ValueError(f"decision must be pass, fail or retry, not {decision!r}")
        with self._tx():
            row = self._db.execute(
                f"SELECT trial_dir FROM runs {_WHERE_INDEX} AND outcome = 'held'", (key, run_index)
            ).fetchone()
            if row is None:
                return False
            if decision == "retry":
                self._db.execute(f"DELETE FROM runs {_WHERE_INDEX}", (key, run_index))
                self._log_retry(key, run_index, reason or "reviewed_retry", "", row[0])
            else:
                reward = reward if reward is not None else (1.0 if decision == "pass" else 0.0)
                self._db.execute(
                    f"UPDATE runs SET outcome = ?, reward = ?, finished_at = ? {_WHERE_INDEX}",
                    (decision, float(reward), time.time(), key, run_index),
                )
        return True

    def record_outcome(
        self, key: str, run_index: int, reward: float, trial_dir: str | None = None,
        metrics: dict | None = None,
    ) -> bool:
        """Write a final pass/fail directly (pass iff reward >= 1.0); also finalizes a held index.

        For automated rules once they exist. Returns False if the index is already final.
        """
        outcome = "pass" if reward >= 1.0 else "fail"
        with self._tx():
            cur = self._db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(scenario_key, run_index) "
                "DO UPDATE SET outcome = excluded.outcome, reward = excluded.reward, "
                "trial_dir = COALESCE(excluded.trial_dir, runs.trial_dir), "
                "metrics_json = excluded.metrics_json, finished_at = excluded.finished_at "
                "WHERE runs.outcome = 'held'",
                (key, run_index, outcome, float(reward), trial_dir,
                 json.dumps(metrics or {}, sort_keys=True), time.time()),
            )
            self._db.execute(f"DELETE FROM leases {_WHERE_INDEX}", (key, run_index))
        return cur.rowcount == 1

    def _log_retry(self, key: str, run_index: int, error_class: str, message: str,
                   trial_dir: str | None) -> int:
        """Inside a transaction: log one retry of an index and free its lease. Returns tries so far."""
        tries = self._db.execute(
            f"SELECT COUNT(*) FROM infra_attempts {_WHERE_INDEX}", (key, run_index)
        ).fetchone()[0] + 1
        self._db.execute(
            "INSERT INTO infra_attempts VALUES (?, ?, ?, ?, ?, ?, ?)",
            (key, run_index, tries, error_class, message[:2000], trial_dir, time.time()),
        )
        self._db.execute(f"DELETE FROM leases {_WHERE_INDEX}", (key, run_index))
        return tries

    def record_infra(
        self, key: str, run_index: int, error_class: str, message: str = "",
        trial_dir: str | None = None,
    ) -> int:
        """Release an index for retry (automated path). Clears a hold, never a final outcome."""
        with self._tx():
            self._db.execute(f"DELETE FROM runs {_WHERE_INDEX} AND outcome = 'held'", (key, run_index))
            return self._log_retry(key, run_index, error_class, message, trial_dir)

    def find_trial(self, trial_dir: str) -> tuple[str, int, int] | None:
        """(scenario_key, run_index, attempt) of an already-logged trial dir, or None."""
        return self._db.execute(
            "SELECT scenario_key, run_index, attempt FROM trials WHERE trial_dir = ?", (trial_dir,)
        ).fetchone()

    def scenario_fields(self, key: str) -> dict:
        """The scenario's fields (task, model, agent, tool mode, settings) as registered."""
        row = self._db.execute(
            "SELECT fields_json FROM scenarios WHERE scenario_key = ?", (key,)).fetchone()
        if row is None:
            raise KeyError(f"unknown scenario {key!r}")
        return json.loads(row[0])

    def scenarios(self) -> list[dict]:
        """Every registered scenario with its fields and summary, ordered by task."""
        rows = self._db.execute("SELECT scenario_key, fields_json FROM scenarios ORDER BY task")
        return [{"key": k, "fields": json.loads(f), **self.summary(k)} for k, f in rows]

    def trial_facts(self, key: str, run_index: int | None = None) -> list[dict]:
        """Raw facts of logged trials (all, or one index), oldest first, for review and reports."""
        sql = "SELECT run_index, attempt, trial_dir, facts_json FROM trials WHERE scenario_key = ?"
        args: tuple = (key,)
        if run_index is not None:
            sql, args = sql + " AND run_index = ?", (key, run_index)
        return [
            {"run_index": i, "attempt": a, "trial_dir": d, "facts": json.loads(f)}
            for i, a, d, f in self._db.execute(sql + " ORDER BY run_index, attempt", args)
        ]

    def release_stale(self, max_age_s: float) -> int:
        """Crash recovery: drop leases older than max_age_s so their indices can be re-leased."""
        cur = self._db.execute("DELETE FROM leases WHERE leased_at < ?", (time.time() - max_age_s,))
        return cur.rowcount

    def freeze(self, key: str) -> None:
        """Mark a scenario finished (quality bar passed). Frozen scenarios lease nothing."""
        self._db.execute(
            "UPDATE scenarios SET frozen_at = ? WHERE scenario_key = ?", (time.time(), key)
        )

    def summary(self, key: str) -> dict:
        """Passed/failed/held/abandoned/leased indices, the next index to run, and what remains."""
        target, frozen_at = self._scenario_row(key)
        rows = list(self._db.execute(
            "SELECT run_index, outcome FROM runs WHERE scenario_key = ? ORDER BY run_index", (key,)
        ))
        passed = [i for i, o in rows if o == "pass"]
        failed = [i for i, o in rows if o == "fail"]
        held = [i for i, o in rows if o == "held"]
        blocked = self._final(key) | self._held(key) | self._leased(key) | self._abandoned(key)
        nxt = next(i for i in range(1, len(blocked) + 2) if i not in blocked)
        return {
            "target": target, "frozen": frozen_at is not None,
            "passed": passed, "failed": failed, "held": held,
            "abandoned": sorted(self._abandoned(key)), "leased": sorted(self._leased(key)),
            "next_index": nxt, "remaining": max(0, target - len(passed) - len(failed)),
        }
