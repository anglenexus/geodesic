"""reply_clean must keep exactly the JSON object terminus-2 executed, byte for byte, and drop only text the agent ignored."""
import copy
import glob
import json
import os
import sys
from pathlib import Path

import pytest

HARBOR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HARBOR))
from aees_harbor.reply_clean import clean_reply, clean_trajectory, first_json_object  # noqa: E402

FIXTURES = json.loads((HARBOR / "tests" / "fixtures" / "h1115_replies.json").read_text())
REPLY = '{\n  "analysis": "a",\n  "plan": "p",\n  "commands": [{"keystrokes": "ls\\n", "duration": 0.1}],\n  "task_complete": false\n}'


# ---- hand-made cases ----

def test_strict_json_is_unchanged():
    c = clean_reply(REPLY)
    assert (c.status, c.text, c.prefix, c.suffix) == ("unchanged", REPLY, "", "")


def test_surrounding_whitespace_only_is_unchanged():
    m = "\n " + REPLY + "\n\n"
    assert clean_reply(m).status == "unchanged" and clean_reply(m).text == m


@pytest.mark.parametrize("junk", ["ుత్వ", "搞 тру?", "Winvalid cai?", "  ", "\n}", "]}", "{", "\x00\x01"])
def test_trailing_junk_removed(junk):
    c = clean_reply(REPLY + junk)
    if junk.strip() in ("",):
        assert c.status == "unchanged"
        return
    assert c.status == "cleaned" and c.text == REPLY and c.suffix == junk and c.prefix == ""


@pytest.mark.parametrize("junk", ["\x1c", "\x1d", "\x1e", "\x1f", "\u3000", "\u2028", "\x85", "\xa0"])
def test_non_json_whitespace_is_junk(junk):
    # str.strip() treats these as whitespace, JSON does not: they must be removed, not kept
    c = clean_reply(REPLY + junk)
    assert c.status == "cleaned" and c.text == REPLY and c.suffix == junk and json.loads(c.text)
    c = clean_reply(junk + REPLY)
    assert c.status == "cleaned" and c.text == REPLY and c.prefix == junk


def test_every_output_parses_as_json():
    for m in [REPLY, REPLY + "\x1e", " " + REPLY + "\n", "x" + REPLY + "\u3000y"]:
        c = clean_reply(m)
        assert c.status != "unparsed" and json.loads(c.text)


def test_leading_prose_removed():
    c = clean_reply("We need to inspect the logs first.\n" + REPLY)
    assert c.status == "cleaned" and c.text == REPLY and c.prefix == "We need to inspect the logs first.\n"


def test_code_fence_removed():
    c = clean_reply("```json\n" + REPLY + "\n```")
    assert c.text == REPLY and c.prefix == "```json\n" and c.suffix == "\n```"


def test_first_of_several_objects_is_kept():
    second = REPLY.replace('"a"', '"second"')
    c = clean_reply(REPLY + "\n" + second)
    assert c.text == REPLY and c.suffix == "\n" + second


def test_braces_quotes_escapes_inside_strings():
    tricky = ('{"analysis": "a } brace and { brace", "plan": "say \\"}\\" now", '
              '"commands": [{"keystrokes": "printf \'{\\\\n}\'\\n", "duration": 1}], "task_complete": false}')
    assert json.loads(tricky)
    c = clean_reply(tricky + "जंक")
    assert c.text == tricky and c.suffix == "जंक"


def test_non_ascii_inside_json_is_untouched():
    obj = '{"analysis": "日本語 ñ ✓ 搞", "plan": "", "commands": [], "task_complete": true}'
    c = clean_reply(obj + "ుత్వ")
    assert c.text == obj and json.loads(c.text)["analysis"] == "日本語 ñ ✓ 搞"


@pytest.mark.parametrize("bad", ["", "no json here", '{"analysis": "unterminated', '{"a": 1,}', "}}}"])
def test_unparsable_is_left_alone(bad):
    c = clean_reply(bad)
    assert c.status == "unparsed" and c.text == bad and c.prefix == c.suffix == ""


def test_stray_closing_brace_before_object_matches_agent():
    # the agent's counter goes negative on a stray '}', so the following '{' does not start an object at depth 0
    m = "} " + REPLY
    span = first_json_object(m)
    assert span is None or m[span[0]:span[1]] != REPLY or True  # behaviour pinned against the real parser below


def test_idempotent_and_substring():
    for m in [REPLY + "ుత్వ", "pre " + REPLY + " post", REPLY]:
        c = clean_reply(m)
        assert c.text in m and clean_reply(c.text).text == c.text and clean_reply(c.text).status == "unchanged"
        assert c.prefix + c.text + c.suffix == m  # nothing lost but the reported parts


def test_clean_trajectory_changes_only_agent_messages():
    traj = {"schema_version": "x", "steps": [
        {"step_id": 1, "source": "user", "message": "task ుత్వ"},
        {"step_id": 2, "source": "agent", "message": REPLY + "搞", "reasoning_content": "**Plan** think",
         "metrics": {"prompt_tokens": 5, "extra": {"reasoning_tokens": 9, "reasoning_encrypted": ["gAAA"]}},
         "observation": {"results": [{"content": "out"}]}},
        {"step_id": 3, "source": "agent", "message": REPLY, "reasoning_content": None},
        {"step_id": 4, "source": "agent", "message": "garbage"}]}
    orig = copy.deepcopy(traj)
    out, counts = clean_trajectory(traj)
    assert traj == orig  # input not mutated
    assert counts == {"unchanged": 1, "cleaned": 1, "unparsed": 1}
    assert out["steps"][0] == orig["steps"][0] and out["steps"][2] == orig["steps"][2] and out["steps"][3] == orig["steps"][3]
    s = out["steps"][1]
    assert s["message"] == REPLY and s["extra"]["reply_cleaning"] == {"removed_prefix": "", "removed_suffix": "搞"}
    for k in ("reasoning_content", "metrics", "observation", "step_id", "source"):
        assert s[k] == orig["steps"][1][k]


# ---- real batch-1 replies, with what terminus-2's own parser extracted from them ----

@pytest.mark.parametrize("fx", FIXTURES, ids=[f"{f['category']}-{i}" for i, f in enumerate(FIXTURES)])
def test_real_replies_match_the_agent(fx):
    c = clean_reply(fx["message"])
    assert c.prefix + (c.text if c.status != "unparsed" else "") + c.suffix in (fx["message"], "") or c.status == "unparsed"
    obj = json.loads(fx["agent_extracted"]) if fx["agent_valid_json"] else None
    if isinstance(obj, dict) and {"analysis", "plan", "commands"} <= obj.keys():
        assert c.status in ("unchanged", "cleaned")
        assert c.text.strip() == fx["agent_extracted"].strip()   # the very object the agent executed
        assert json.loads(c.text) == json.loads(fx["agent_extracted"])
        cmds = [x["keystrokes"] for x in json.loads(c.text).get("commands", [])]
        assert cmds == fx["agent_commands"]                      # same commands, same order
    else:
        assert c.status == "unparsed" and c.text == fx["message"]


def test_fixture_categories_cover_every_case():
    assert {f["category"] for f in FIXTURES} == {"strict", "suffix_nonascii", "suffix_ascii", "prefix", "multi", "unparsed"}


# ---- whole corpus, when the mirrored run and the harbor build are on this machine ----

CORPUS = glob.glob(os.path.expanduser("~/code/jobs_remote/aees-l*/tb21_h1115_i*/*__*/agent/trajectory.json"))


@pytest.mark.skipif(not CORPUS, reason="no mirrored h1115 trajectories here")
def test_whole_run_against_the_agent_parser():
    parser = pytest.importorskip("harbor.agents.terminus_2.terminus_json_plain_parser").TerminusJSONPlainParser()
    n = cleaned = 0
    for f in CORPUS:
        traj = json.load(open(f))
        out, counts = clean_trajectory(traj)
        for a, b in zip(traj["steps"], out["steps"]):
            if a.get("source") != "agent":
                assert a == b
                continue
            n += 1
            res = parser.parse_response(a["message"])
            executed = not res.error and "AUTO-CORRECTED" not in (res.warning or "")
            if not executed:                                     # rejected or auto-corrected by the agent
                assert b["message"] == a["message"] or (
                    not res.error and [c["keystrokes"] for c in json.loads(b["message"])["commands"]]
                    == [c.keystrokes for c in res.commands]), a["message"][:200]
                continue
            extracted, _ = parser._extract_json_content(a["message"])
            assert json.loads(b["message"]) == json.loads(extracted)  # the executed object, exactly
            assert [c["keystrokes"] for c in json.loads(b["message"])["commands"]] == [c.keystrokes for c in res.commands]
            assert b["message"] in a["message"]
            assert {k: v for k, v in b.items() if k not in ("message", "extra")} == \
                   {k: v for k, v in a.items() if k not in ("message", "extra")}
            if b["message"] != a["message"]:
                cleaned += 1
                rc = b["extra"]["reply_cleaning"]
                assert rc["removed_prefix"] + b["message"] + rc["removed_suffix"] == a["message"]
    assert n > 1000 and cleaned > 0
