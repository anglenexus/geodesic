"""Tests for sft/rollout.py with the real Qwen3.6 tokenizer (skipped if it isn't in the HF cache).

Fetch once: uvx --from huggingface_hub hf download Qwen/Qwen3.6-27B tokenizer.json tokenizer_config.json \
    vocab.json merges.txt chat_template.jinja config.json
"""
import json
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from sft import rollout as ro

transformers = pytest.importorskip("transformers")
try:
    TOK = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B", local_files_only=True)
except OSError:
    pytest.skip("Qwen3.6-27B tokenizer not in the HF cache", allow_module_level=True)

OPENER = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
EXPORTED = Path(__file__).parent.parent / "data" / "sft_qwen35_27b.jsonl"
WORDS = ["ls -la", "cat /app/x.py", '{"keystrokes": "ls\\n", "duration": 0.1}', "error:", "ok", "日本語", "<tag>",
         "\n", "  ", "émoji 🙂", "\t", "```bash", "```", "<think>", "<|im_start|>", "<|im_end|>", "\\n", "$ "]
# "</think>" is left out on purpose: see test_reply_containing_end_think_fails_loudly


def random_text(rng: random.Random, allow_empty: bool = True) -> str:
    text = "".join(rng.choice(WORDS) + rng.choice([" ", "", "\n"]) for _ in range(rng.randint(0, 20)))
    return text if (allow_empty or text.strip()) else "x"


def random_chat(rng: random.Random, p_masked: float = 0.3) -> list[dict]:
    """user, (assistant, user)*, assistant; replies stripped like the exporter's, some with step_loss_mask 0."""
    chat = [{"role": "user", "content": random_text(rng)}]
    for _ in range(rng.randint(1, 8)):
        reply = {"role": "assistant", "content": random_text(rng, allow_empty=False).strip() or "x"}
        if rng.random() < p_masked:
            reply["step_loss_mask"] = 0
        chat += [reply, {"role": "user", "content": random_text(rng)}]
    chat.pop()
    return chat


def trained(msg: dict) -> bool:
    return msg["role"] == "assistant" and msg.get("step_loss_mask", 1) == 1


def spans(mask: list[int]) -> list[tuple[int, int]]:
    """[start, end) of each run of 1s."""
    out, start = [], None
    for i, m in enumerate(mask + [0]):
        if m and start is None:
            start = i
        elif not m and start is not None:
            out.append((start, i))
            start = None
    return out


# --- render_qwen_format ---

def test_render_example_from_the_docstring():
    chat = [{"role": "user", "content": "ls"}, {"role": "assistant", "content": "done"}]
    assert ro.render_qwen_format(TOK, chat) == "<|im_start|>user\nls<|im_end|>\n" + OPENER + "done<|im_end|>\n"
    assert ro.render_qwen_format(TOK, chat[:1], add_generation_prompt=True) == "<|im_start|>user\nls<|im_end|>\n" + OPENER


@pytest.mark.parametrize("seed", range(200))
def test_render_keeps_every_think_block_and_is_append_only(seed):
    chat = random_chat(random.Random(seed))
    texts = [ro.render_qwen_format(TOK, chat[: i + 1]) for i in range(len(chat))]
    assert all(b.startswith(a) for a, b in zip(texts, texts[1:]))
    assert texts[-1].count(OPENER) == sum(m["role"] == "assistant" for m in chat)


def test_native_settings_would_break_append_only():
    """Why this file exists: without preserve_thinking, earlier turns lose their think block."""
    chat = random_chat(random.Random(0))
    chat = chat + [{"role": "user", "content": "next"}]
    native = TOK.apply_chat_template(chat, tokenize=False, enable_thinking=False)
    assert native.count(OPENER) < sum(m["role"] == "assistant" for m in chat)


# --- render_qwen_format_with_loss_mask ---

@pytest.mark.parametrize("seed", range(200))
def test_tokens_equal_tokenizing_the_whole_text(seed):
    chat = random_chat(random.Random(seed))
    tokens, mask = ro.render_qwen_format_with_loss_mask(TOK, chat)
    assert len(tokens) == len(mask) and set(mask) <= {0, 1}
    assert tokens == TOK.encode(ro.render_qwen_format(TOK, chat), add_special_tokens=False)


@pytest.mark.parametrize("seed", range(200))
def test_each_trained_span_is_one_reply_plus_im_end(seed):
    chat = random_chat(random.Random(seed))
    tokens, mask = ro.render_qwen_format_with_loss_mask(TOK, chat)
    got = [TOK.decode(tokens[a:b]) for a, b in spans(mask)]
    assert got == [m["content"] + ro.IM_END for m in chat if trained(m)]


@pytest.mark.parametrize("seed", range(200))
def test_each_trained_span_follows_an_opener_and_precedes_a_newline(seed):
    chat = random_chat(random.Random(seed))
    tokens, mask = ro.render_qwen_format_with_loss_mask(TOK, chat)
    opener_ids = TOK.encode(OPENER, add_special_tokens=False)
    for a, b in spans(mask):
        assert tokens[a - len(opener_ids):a] == opener_ids
        assert TOK.decode(tokens[b:b + 1]) == "\n"


def test_reply_containing_end_think_fails_loudly():
    """Qwen's template treats text before "</think>" in a reply as reasoning and moves it into the think block."""
    chat = [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a </think> b"}]
    with pytest.raises(AssertionError, match="odd assistant turn"):
        ro.render_qwen_format_with_loss_mask(TOK, chat)


def test_all_masked_chat_has_no_trained_tokens():
    chat = random_chat(random.Random(1), p_masked=1.0)
    _, mask = ro.render_qwen_format_with_loss_mask(TOK, chat)
    assert sum(mask) == 0


def test_no_masked_chat_trains_every_reply():
    chat = random_chat(random.Random(2), p_masked=0.0)
    tokens, mask = ro.render_qwen_format_with_loss_mask(TOK, chat)
    assert len(spans(mask)) == sum(m["role"] == "assistant" for m in chat)


def check_chats(tok, rows: list[dict]) -> None:
    """The loss-mask check: tokens match the whole text, and trained spans are exactly the trained replies."""
    for row in rows:
        chat = row["messages"]
        tokens, mask = ro.render_qwen_format_with_loss_mask(tok, chat)
        assert tokens == tok.encode(ro.render_qwen_format(tok, chat), add_special_tokens=False)
        assert [tok.decode(tokens[a:b]) for a, b in spans(mask)] == [m["content"] + ro.IM_END for m in chat if trained(m)]


@pytest.mark.skipif(not EXPORTED.exists(), reason="run sft/export_trajectories.py first")
def test_every_exported_chat():
    check_chats(TOK, [json.loads(line) for line in EXPORTED.read_text().splitlines()])


@pytest.mark.skipif(not EXPORTED.exists(), reason="run sft/export_trajectories.py first")
def test_every_exported_chat_with_miles_tokenizer():
    """On the box, inside the Miles container: the same check with the trainer's own tokenizer loader.

    MODEL_DIR defaults to /root/models/Qwen3.6-27B. Skipped on the Mac (no Miles).
    """
    loader = pytest.importorskip("miles.utils.processing_utils")
    model_dir = Path(os.environ.get("MODEL_DIR", "/root/models/Qwen3.6-27B"))
    if not model_dir.exists():
        pytest.skip(f"{model_dir} not found")
    tok = loader.load_tokenizer(str(model_dir), trust_remote_code=True)
    check_chats(tok, [json.loads(line) for line in EXPORTED.read_text().splitlines()])


# --- generate_rollout ---

@pytest.fixture
def fake_miles(monkeypatch):
    """Stand-in for miles.utils.processing_utils.load_tokenizer (Miles isn't installed on the host)."""
    loader = types.ModuleType("miles.utils.processing_utils")
    loader.load_tokenizer = lambda path, **kw: TOK
    for name, mod in [("miles", types.ModuleType("miles")), ("miles.utils", types.ModuleType("miles.utils")),
                      ("miles.utils.processing_utils", loader)]:
        monkeypatch.setitem(sys.modules, name, mod)


def run_rollout(chats: list[list[dict]], **kw):
    batch = [[SimpleNamespace(prompt=c)] for c in chats]
    args = SimpleNamespace(hf_checkpoint="Qwen/Qwen3.6-27B", rollout_batch_size=len(chats))
    return ro.generate_rollout(args, 0, SimpleNamespace(get_samples=lambda n: batch[:n]), **kw)


@pytest.mark.parametrize("seed", range(50))
def test_generate_rollout_fills_samples(fake_miles, seed):
    rng = random.Random(seed)
    chats = [c for c in (random_chat(rng) for _ in range(8)) if any(trained(m) for m in c)]
    out = run_rollout(chats)
    assert len(out) == len(chats)
    for (s,), chat in zip(out, chats):
        tokens, mask = ro.render_qwen_format_with_loss_mask(TOK, chat)
        assert s.tokens == tokens and s.reward == 0
        assert len(s.loss_mask) == s.response_length and s.loss_mask[0] == 1
        assert s.loss_mask == mask[len(tokens) - s.response_length:]
        assert sum(mask[:len(tokens) - s.response_length]) == 0      # nothing trained is cut off


def test_generate_rollout_rejects_evaluation(fake_miles):
    with pytest.raises(AssertionError):
        run_rollout([random_chat(random.Random(0))], evaluation=True)


def test_generate_rollout_fails_loudly_on_a_chat_with_nothing_to_train(fake_miles):
    with pytest.raises(ValueError):
        run_rollout([random_chat(random.Random(1), p_masked=1.0)])
