"""Clean terminus-2 JSON replies for training data, without changing what the agent did.

GPT-5.6 Luna (high reasoning, reasoning sent back; run h1115) often ends a reply with stray text after the JSON object,
mostly non-ASCII fragments ('}ుత్వ', '}搞 тру?'): 14.4% of batch-1 steps, against 0.7% at medium reasoning. A few
replies also start with prose or hold a second JSON object. The agent never saw that text as an action: terminus-2's
parser (harbor-miles terminus_json_plain_parser._extract_json_content) takes the FIRST complete top-level {...},
with string- and escape-aware brace counting, and ignores everything before and after it.

clean_reply() keeps exactly that object as an exact substring of the original (nothing is re-serialized, so key
order, whitespace, escapes and non-ASCII text inside the JSON are byte-identical) and returns what it removed. A reply
whose first object is not valid JSON or not a complete reply (no "analysis", "plan" and "commands" list; the agent
rejected such steps, or auto-corrected them into something else) is not changed: status "unparsed".
clean_trajectory() applies it to every agent step of a trajectory.json and copies everything else unchanged
(reasoning_content, metrics, observations). The raw files on disk are never modified.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass


REQUIRED = {"analysis", "plan", "commands"}  # terminus-2 rejects a reply without them
JSON_WS = " \t\n\r"  # only these; str.strip() would also drop e.g. \x1c-\x1f and \u3000, which break json.loads


@dataclass(frozen=True)
class CleanedReply:
    text: str        # the JSON object the agent executed (or the original message when status is "unparsed")
    prefix: str      # removed text before the object ("" if none)
    suffix: str      # removed text after the object ("" if none)
    status: str      # "unchanged" | "cleaned" | "unparsed"


def first_json_object(message: str) -> tuple[int, int] | None:
    """[start, end) of the first complete top-level {...}, scanning like terminus-2's parser."""
    start, depth, in_string, escape = -1, 0, False, False
    for i, ch in enumerate(message):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":  # like the agent's parser, a stray '}' may take the count below zero
            depth -= 1
            if depth == 0 and start != -1:
                return start, i + 1
    return None


def clean_reply(message: str) -> CleanedReply:
    span = first_json_object(message)
    if span is None:
        return CleanedReply(message, "", "", "unparsed")
    start, end = span
    obj = message[start:end]
    try:
        data = json.loads(obj)
    except json.JSONDecodeError:
        return CleanedReply(message, "", "", "unparsed")
    if not (isinstance(data, dict) and REQUIRED <= data.keys() and isinstance(data["commands"], list)):
        return CleanedReply(message, "", "", "unparsed")  # not what the agent executed: leave it as it is
    prefix, suffix = message[:start], message[end:]
    if not prefix.strip(JSON_WS) and not suffix.strip(JSON_WS):
        return CleanedReply(message, "", "", "unchanged")  # JSON whitespace around the object is kept as it was
    return CleanedReply(obj, prefix, suffix, "cleaned")


def clean_trajectory(trajectory: dict) -> tuple[dict, dict]:
    """A cleaned copy of a trajectory.json dict and counts by status. Only agent steps' `message` changes; each
    changed step gets extra["reply_cleaning"] = {"removed_prefix", "removed_suffix"} so the cleaning is traceable."""
    out = copy.deepcopy(trajectory)
    counts = {"unchanged": 0, "cleaned": 0, "unparsed": 0}
    for step in out.get("steps", []):
        if step.get("source") != "agent" or not isinstance(step.get("message"), str):
            continue
        c = clean_reply(step["message"])
        counts[c.status] += 1
        if c.status == "cleaned":
            step["message"] = c.text
            step["extra"] = {**(step.get("extra") or {}),
                             "reply_cleaning": {"removed_prefix": c.prefix, "removed_suffix": c.suffix}}
    return out, counts
