"""Excludes infrastructure failures from GRPO groups one attempt at a time (PLAN.md §8 item 6).

An attempt is excluded only when the policy cannot have caused its failure; every other attempt keeps
its reward (the verifier's, or 0). An excluded attempt gets loss mask 0 and stays out of its group's
mean and std, so the other attempts' advantages are computed among themselves.
Input: Miles samples, one group of --n-samples-per-prompt after another. Each sample.metadata holds the
agent server's reply {"reward", "exit_status", "agent_metrics": {..., "exception_type"}}
(grpo/harbor_parser.patch), or has no "exit_status" when the agent server couldn't be reached.
    --custom-reward-post-process-path grpo.infra_filter.post_process_rewards
Attempts with no model call never get here: Miles drops their whole group (apply_aborted_filter).
"""
import logging
from collections import Counter

logger = logging.getLogger(__name__)
SERVER_ERRORS = {"TaskNotFound", "InvalidInstanceId", "InvalidSessionServerId", "Flushed", "ImportError"}
SETUP_EXCEPTIONS = {"EnvironmentStartTimeoutError", "AgentSetupTimeoutError"}   # before the agent's first step
MIN_VALID = 4          # fewer valid attempts left in a group: mask the whole group (std too noisy)
MAX_EXCLUDED = 0.10    # more of a step excluded: stop the run (an outage, not noise)


def infra_cause(metadata: dict) -> str | None:
    """Why this attempt is an infrastructure failure, or None if its reward stands."""
    status = metadata.get("exit_status")
    if status is None:
        return "AgentServerUnreachable"                # swe_agent_function.run returned None
    if status.startswith("Error:") or status in SERVER_ERRORS:
        return status                                  # the agent server failed, not the trial
    exc = (metadata.get("agent_metrics") or {}).get("exception_type", "")
    return exc if exc in SETUP_EXCEPTIONS else None    # AgentError, TimeLimitExceeded, ... keep their reward


def post_process_rewards(args, samples) -> tuple[list[float], list[float]]:
    """Miles' reward post-process hook: (raw rewards, GRPO advantages), one per sample. Same as Miles'
    default, except excluded attempts get advantage 0 and loss mask 0, and each group's mean and std
    come from its remaining attempts only."""
    from miles.ray.rollout.train_data_conversion import _normalize_rewards_by_rollout   # Miles' GRPO math
    n, raw = args.n_samples_per_prompt, [s.get_reward_value(args) for s in samples]
    assert len(samples) % n == 0, f"{len(samples)} samples are not whole groups of {n}"
    adv, causes = [0.0] * len(samples), Counter()
    for g in range(0, len(samples), n):
        cause = {i: infra_cause(samples[i].metadata) for i in range(g, g + n)}
        valid = [i for i, c in cause.items() if c is None]
        if len(valid) < MIN_VALID:
            cause, valid = {i: c or "TooFewValid" for i, c in cause.items()}, []
        for i, c in cause.items():
            if c:
                samples[i].remove_sample = True        # Miles zeroes its loss mask
                causes[c] += 1
        if valid:
            norm = _normalize_rewards_by_rollout(args, [samples[i] for i in valid], [raw[i] for i in valid], [len(valid)])
            for i, a in zip(valid, norm):
                adv[i] = a
    logger.info(f"infra_filter: excluded {sum(causes.values())}/{len(samples)} attempts: {dict(causes)}")
    if sum(causes.values()) > MAX_EXCLUDED * len(samples):
        raise RuntimeError(f"over {MAX_EXCLUDED:.0%} of the step excluded: {dict(causes)} (outage?)")
    return raw, adv
