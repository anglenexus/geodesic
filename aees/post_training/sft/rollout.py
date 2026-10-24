"""Miles SFT rollout: renders terminus-2 chats token for token as the GRPO session builds them.

Miles' built-in sft_rollout renders with no chat-template kwargs. With Qwen3.6 that drops the empty
think block from earlier assistant turns and puts the last turn's block inside the trained span. In
GRPO (thinking off, --tito-model qwen36 with preserve_thinking) every assistant turn is
    <|im_start|>assistant\n<think>\n\n</think>\n\n   context: the generation prompt
    reply<|im_end|>                                    trained: what the model generates
    \n                                                 context: appended by the session
This rollout builds the same stream. Assistant messages with step_loss_mask 0 are context only.

    --rollout-function-path sft.rollout.generate_rollout --input-key messages --loss-type sft_loss

References:
    Qwen3.6 chat template:   https://huggingface.co/Qwen/Qwen3.6-27B/blob/main/chat_template.jinja
    apply_chat_template:     https://huggingface.co/docs/transformers/main/en/chat_templating
    Miles built-in rollout:  miles/rollout/sft_rollout.py (what this file replaces)
    Miles GRPO token stream: miles/utils/chat_template_utils/tito_tokenizer.py (Qwen36TITOTokenizer)
"""
TEMPLATE_KWARGS = {"enable_thinking": False, "preserve_thinking": True}   # as in GRPO
IM_END = "<|im_end|>"


def render_qwen_format(tokenizer, messages: list[dict], **kw) -> str:
    """Messages -> the plain text the model reads, in Qwen's chat format (ChatML), thinking off.

    Each message becomes "<|im_start|>{role}\\n{content}<|im_end|>\\n". Assistant content is preceded by an
    empty think block "<think>\\n\\n</think>\\n\\n". add_generation_prompt=True appends the next turn's
    opener "<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n", i.e. the prompt the model continues from.
    Example: [{"role": "user", "content": "ls"}, {"role": "assistant", "content": "done"}] ->
    "<|im_start|>user\\nls<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\ndone<|im_end|>\\n"
    The format is defined by the model's chat_template.jinja (see References above), not by Miles.
    """
    return tokenizer.apply_chat_template(messages, tokenize=False, **TEMPLATE_KWARGS, **kw)


def render_qwen_format_with_loss_mask(tokenizer, messages: list[dict]) -> tuple[list[int], list[int]]:
    """Messages -> token ids plus a same-length 0/1 mask: 1 = the assistant's own words (trained), 0 = context.

    Goes message by message: each message adds a piece of chat text, tokenized and appended. For an assistant
    message, "reply<|im_end|>" gets 1, and its opening "<|im_start|>assistant\\n<think>...</think>" and the final "\\n" get 0.
    """
    tokens, mask, before = [], [], ""
    for i, msg in enumerate(messages):
        after = render_qwen_format(tokenizer, messages[: i + 1])
        assert after.startswith(before), f"template rewrote earlier turns at message {i}"
        new_text = after[len(before):]
        pieces = [(new_text, 0)]                       # user message: all context
        if msg["role"] == "assistant":
            opener = render_qwen_format(tokenizer, messages[:i], add_generation_prompt=True)[len(before):]
            assert new_text.startswith(opener) and new_text.endswith(IM_END + "\n"), f"odd assistant turn {i}"
            trained = int(msg.get("step_loss_mask", 1) == 1)   # 0 for copied hand-offs and parse errors
            pieces = [(opener, 0), (new_text[len(opener):-1], trained), ("\n", 0)]
        for text, m in pieces:
            ids = tokenizer.encode(text, add_special_tokens=False)
            tokens += ids
            mask += [m] * len(ids)
        before = after
    return tokens, mask


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    """Miles calls this once per training step: take the next batch of chats, fill in tokens and loss mask.

    Miles treats everything from the first trained token to the end as the "response"; its loss mask
    covers only that span. The tokens before it (system prompt, task, first opener) stay in the input.
    """
    assert not evaluation, "SFT has no evaluation rollout"
    from miles.utils.processing_utils import load_tokenizer       # cached by Miles after the first call
    tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
    samples = data_buffer.get_samples(args.rollout_batch_size)    # list of [Sample], one chat each
    for (sample,) in samples:
        tokens, mask = render_qwen_format_with_loss_mask(tokenizer, sample.prompt)   # prompt = the messages
        first = mask.index(1)                                     # the exporter keeps only chats with a trained reply
        sample.tokens, sample.response_length = tokens, len(tokens) - first
        sample.loss_mask, sample.reward = mask[first:], 0
    return samples
