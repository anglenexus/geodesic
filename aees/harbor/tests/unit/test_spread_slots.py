"""spread_h1115.slots: per-box slot counts that make the box's last job finish as early as possible."""
import itertools
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import spread_h1115 as s  # noqa: E402


def longest(share, w, n):
    return max(share[t] * w.get(t, 10.0) / n[t] for t in share)


def brute(share, w, per_box):
    """Best possible longest job over every allocation (small cases only)."""
    ts = list(share)
    best = None
    for combo in itertools.product(*[range(1, min(share[t], per_box) + 1) for t in ts]):
        if sum(combo) <= per_box:
            v = longest(share, w, dict(zip(ts, combo)))
            best = v if best is None or v < best else best
    return best


def test_batch2_box_is_balanced():
    # aees-l1's batch-2 jobs on 2026-11-15 (attempts, batch-1 median minutes)
    share = {"count-dataset-tokens": 219, "log-summary-date-ranges": 111, "db-wal-recovery": 92, "sqlite-with-gcov": 88,
             "sanitize-git-repo": 70, "largest-eigenval": 63, "dna-insert": 56, "constraints-scheduling": 47,
             "crack-7z-hash": 46, "feal-differential-cryptanalysis": 46, "polyglot-rust-c": 46, "write-compressor": 47,
             "cobol-modernization": 45, "polyglot-c-py": 45, "vulnerable-secret": 45, "custom-memory-heap-crash": 42,
             "feal-linear-cryptanalysis": 43, "regex-log": 43, "code-from-image": 42, "multi-source-data-merger": 41}
    w = {"count-dataset-tokens": 4.7, "log-summary-date-ranges": 1.8, "db-wal-recovery": 2.5, "sqlite-with-gcov": 4.0,
         "sanitize-git-repo": 4.1, "largest-eigenval": 6.8, "dna-insert": 3.6, "constraints-scheduling": 2.0,
         "crack-7z-hash": 6.1, "feal-differential-cryptanalysis": 3.9, "polyglot-rust-c": 3.3, "write-compressor": 6.8,
         "cobol-modernization": 3.2, "polyglot-c-py": 2.5, "vulnerable-secret": 2.1, "custom-memory-heap-crash": 3.5,
         "feal-linear-cryptanalysis": 4.5, "regex-log": 2.3, "code-from-image": 3.7, "multi-source-data-merger": 1.8}
    n = s.slots(share, w, 40)
    assert sum(n.values()) == 40 and all(n[t] >= 1 for t in n)
    packed = sum(share[t] * w[t] for t in share) / 40          # 119 min if perfectly divisible
    assert longest(share, w, n) <= 1.6 * packed                 # was 3.83 h = 1.9x packed with the old rule
    assert longest(share, w, n) < 3.0 * 60


def test_optimal_against_brute_force():
    rng = random.Random(3)
    for _ in range(300):
        k = rng.randint(1, 4)
        share = {f"t{i}": rng.randint(1, 12) for i in range(k)}
        w = {t: rng.choice([1.0, 2.5, 4.0, 7.0]) for t in share}
        per = rng.randint(k, 10)
        n = s.slots(share, w, per)
        assert sum(n.values()) <= per and all(1 <= n[t] <= max(1, share[t]) for t in share)
        if sum(share.values()) > per:
            assert longest(share, w, n) == pytest.approx(brute(share, w, per))


def test_limits():
    assert s.slots({"a": 3, "b": 2}, {}, 40) == {"a": 3, "b": 2}           # everything fits: all start at once
    n = s.slots({"a": 1000, "b": 1000}, {"a": 1.0, "b": 3.0}, 40)
    assert n == {"a": 10, "b": 30}                                          # proportional to work
    n = s.slots({"a": 2, "b": 500}, {}, 40)
    assert n["a"] <= 2 and sum(n.values()) == 40                            # never more slots than attempts
    with pytest.raises(ValueError):
        s.slots({f"t{i}": 5 for i in range(41)}, {}, 40)                    # can't give 41 jobs a slot each


def test_random_invariants():
    rng = random.Random(1)
    for _ in range(2000):
        share = {f"t{i}": rng.randint(1, 400) for i in range(rng.randint(1, 20))}
        w = {t: rng.uniform(1, 30) for t in share}
        n = s.slots(share, w, 40)
        assert sum(n.values()) <= 40 and all(1 <= n[t] <= share[t] for t in share)
        if sum(share.values()) >= 40:
            assert sum(n.values()) == 40
