"""Random-data tests for grpo/infra_filter.py, with a stand-in for Miles' GRPO normalization."""
import random
import statistics
import sys
import types
from types import SimpleNamespace

import pytest

from grpo import infra_filter as inf

INFRA = [  # metadata -> the cause infra_cause must return
    ({}, "AgentServerUnreachable"),
    ({"exit_status": "Error: OSError"}, "Error: OSError"),
    *[({"exit_status": s}, s) for s in sorted(inf.SERVER_ERRORS)],
    *[({"exit_status": "AgentError", "agent_metrics": {"exception_type": e}}, e) for e in sorted(inf.SETUP_EXCEPTIONS)],
    ({"exit_status": "TimeLimitExceeded", "agent_metrics": {"exception_type": "EnvironmentStartTimeoutError"}},
     "EnvironmentStartTimeoutError"),
]
KEPT = [  # policy outcomes: reward stands
    {"exit_status": "Submitted", "agent_metrics": {"exception_type": ""}},
    {"exit_status": "Submitted"},
    {"exit_status": "TimeLimitExceeded", "agent_metrics": {"exception_type": "AgentTimeoutError"}},
    {"exit_status": "TimeLimitExceeded", "agent_metrics": {"exception_type": "VerifierTimeoutError"}},
    {"exit_status": "AgentError", "agent_metrics": {"exception_type": "ContextLengthExceededError"}},
    {"exit_status": "AgentError", "agent_metrics": {"exception_type": "RuntimeError"}},
    {"exit_status": "SequenceLengthLimitExceeded", "agent_metrics": {}},
    {"exit_status": "Unknown", "agent_metrics": None},
]


def miles_normalize(raw: list[float]) -> list[float]:
    """Miles' GRPO normalization for one group (train_data_conversion._normalize_rewards_by_rollout)."""
    mean = sum(raw) / len(raw)
    std = statistics.stdev(raw) if len(raw) > 1 else 0.0
    return [(r - mean) / (std + 1e-6) if std > 0 else r - mean for r in raw]


@pytest.fixture
def fake_miles(monkeypatch):
    calls = []

    def normalize(args, samples, raw, sizes):
        assert sizes == [len(samples)] == [len(raw)]
        calls.append(list(raw))
        return miles_normalize(raw)

    mod = types.ModuleType("miles.ray.rollout.train_data_conversion")
    mod._normalize_rewards_by_rollout = normalize
    for name in ["miles", "miles.ray", "miles.ray.rollout"]:
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return calls


class FakeSample:
    def __init__(self, reward, metadata):
        self.reward, self.metadata, self.remove_sample = reward, metadata, False

    def get_reward_value(self, args):
        return self.reward


@pytest.mark.parametrize("metadata,cause", INFRA)
def test_infra_causes(metadata, cause):
    assert inf.infra_cause(metadata) == cause


@pytest.mark.parametrize("metadata", KEPT)
def test_policy_outcomes_keep_their_reward(metadata):
    assert inf.infra_cause(metadata) is None


@pytest.mark.parametrize("seed", range(200))
def test_groups_normalized_over_remaining_attempts(fake_miles, monkeypatch, seed):
    rng = random.Random(seed)
    monkeypatch.setattr(inf, "MAX_EXCLUDED", 1.0)             # the outage stop has its own test
    n, groups = rng.choice([2, 4, 8, 16]), rng.randint(1, 25)
    p_infra = rng.choice([0.0, 0.05, 0.3, 0.7])
    samples = [FakeSample(float(rng.random() < 0.4), rng.choice(INFRA)[0] if rng.random() < p_infra
                          else rng.choice(KEPT)) for _ in range(n * groups)]
    raw, adv = inf.post_process_rewards(SimpleNamespace(n_samples_per_prompt=n), samples)

    assert raw == [s.reward for s in samples]
    for g in range(0, len(samples), n):
        group = samples[g:g + n]
        valid = [i for i, s in enumerate(group) if inf.infra_cause(s.metadata) is None]
        if len(valid) < inf.MIN_VALID:
            valid = []
        expected = dict(zip(valid, miles_normalize([group[i].reward for i in valid]))) if valid else {}
        for i, s in enumerate(group):
            assert s.remove_sample == (i not in expected)
            assert adv[g + i] == pytest.approx(expected.get(i, 0.0))
    assert len(fake_miles) == sum(1 for g in range(0, len(samples), n)
                                  if sum(inf.infra_cause(s.metadata) is None for s in samples[g:g + n]) >= inf.MIN_VALID)


def test_group_with_too_few_valid_is_masked_whole(fake_miles):
    infra = [FakeSample(0.0, {}) for _ in range(5)]
    kept = [FakeSample(1.0, {"exit_status": "Submitted"}) for _ in range(3)]
    passing = [FakeSample(float(i % 2), {"exit_status": "Submitted"}) for i in range(8 * 7)]
    samples = infra + kept + passing                          # 8 of 64 excluded = 12.5% ...
    with pytest.raises(RuntimeError, match="outage"):
        inf.post_process_rewards(SimpleNamespace(n_samples_per_prompt=8), samples)
    samples = infra + kept + passing + [FakeSample(1.0, {"exit_status": "Submitted"}) for _ in range(16)]
    raw, adv = inf.post_process_rewards(SimpleNamespace(n_samples_per_prompt=8), samples)   # 8 of 80 = 10%: ok
    assert all(s.remove_sample for s in samples[:8]) and adv[:8] == [0.0] * 8
    assert not any(s.remove_sample for s in samples[8:])


def test_outage_stop_threshold(fake_miles):
    args = SimpleNamespace(n_samples_per_prompt=10)
    ok = [FakeSample(0.0, {})] + [FakeSample(float(i % 2), {"exit_status": "Submitted"}) for i in range(9)]
    inf.post_process_rewards(args, ok)                        # 1 of 10 = 10%: continues
    bad = [FakeSample(0.0, {}), FakeSample(0.0, {})] + ok[1:9]
    with pytest.raises(RuntimeError, match="outage"):
        inf.post_process_rewards(args, bad)                   # 2 of 10 = 20%: stops


def test_partial_group_is_rejected(fake_miles):
    with pytest.raises(AssertionError, match="whole groups"):
        inf.post_process_rewards(SimpleNamespace(n_samples_per_prompt=8),
                                 [FakeSample(0.0, {"exit_status": "Submitted"}) for _ in range(12)])
