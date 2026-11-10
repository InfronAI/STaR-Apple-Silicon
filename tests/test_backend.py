import pytest
import torch

from star.backend import TorchBackend, collate, encode_record, loss_sum
from star.config import Config
from star.smoke import create_tiny_model


@pytest.fixture
def backend(tmp_path):
    create_tiny_model(tmp_path / "base")
    config = Config(
        device="cpu",
        local_files_only=True,
        few_shot=0,
        max_seq_length=128,
        max_new_tokens=20,
        batch_size=1,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
    )
    model = TorchBackend(str(tmp_path / "base"), config, training=True)
    yield model
    model.close()


def test_prompt_and_padding_have_zero_loss_gradient(backend):
    record = {"prompt": "Question: 1+1?\nAnswer:", "completion": " Think: 1+1=2. (a)."}
    row = encode_record(backend.tokenizer, record, 128)
    short = encode_record(backend.tokenizer, {"prompt": "Q:", "completion": " a"}, 128)
    batch = collate([row, short], backend.tokenizer.pad_token_id, "cpu")
    logits = torch.randn(2, batch["labels"].shape[1], len(backend.tokenizer), requires_grad=True)
    loss_sum(logits, batch["labels"], 0).backward()
    masked = batch["labels"][:, 1:] == -100
    assert torch.all(logits.grad[:, :-1][masked] == 0)
    assert logits.grad[:, :-1][~masked].abs().sum() > 0
    assert row["labels"][-1] == backend.tokenizer.eos_token_id
    assert encode_record(backend.tokenizer, record, 4) is None


def test_real_training_tail_and_checkpoint_roundtrip(backend, tmp_path):
    records = [{"prompt": "Q: 1+1?\nA:", "completion": " Two (a)."}] * 3
    before = next(backend.model.parameters()).detach().clone()
    metrics = backend.train(records)
    assert metrics["optimizer_steps"] == 1  # Short accumulation group is not discarded.
    assert metrics["examples_seen"] == 3
    assert not torch.equal(before, next(backend.model.parameters()))
    output = backend.generate([records[0]["prompt"]])
    backend.save(tmp_path / "trained")
    restored = TorchBackend(str(tmp_path / "trained"), backend.config)
    try:
        assert restored.generate([records[0]["prompt"]]) == output
    finally:
        restored.close()


def test_question_is_never_truncated_to_fit(backend):
    prompt = backend.fit_prompt("Q: short\nA:", ["very long" * 100])
    assert prompt == "Q: short\nA:"
    with pytest.raises(ValueError, match="Question alone"):
        backend.fit_prompt("long" * 100, [])


def test_generation_has_no_segment_embedding_and_handles_left_padding(backend, monkeypatch):
    # A generic fast tokenizer returns token_type_ids; GPT-J interprets them as
    # additional token embeddings, unlike training. Generation must discard them.
    prompts = ["Q:\nA:", "Q: a longer question\nA:"]
    original_generate = backend.model.generate

    def checked_generate(**kwargs):
        assert "token_type_ids" not in kwargs
        return original_generate(**kwargs)

    monkeypatch.setattr(backend.model, "generate", checked_generate)
    backend.config.generation_batch_size = 2
    together = backend.generate(prompts)
    separate = [backend.generate([prompt])[0] for prompt in prompts]
    assert together == separate
    tokens = backend.tokenizer(prompts[0], return_tensors="pt", return_token_type_ids=False)
    with torch.inference_mode():
        expected = backend.model(**tokens).logits[0, -1].argmax().item()
    # First generated token should come from the exact same forward input as training.
    with torch.inference_mode():
        output = backend.model.generate(**tokens, max_new_tokens=1, do_sample=False)
    assert output[0, -1].item() == expected


@pytest.mark.parametrize("completion", [" Count two.\n#### 2", "<scratch>\n1 + 1\n</scratch>\n2"])
def test_multiline_task_completions_train(backend, completion):
    backend.config.max_steps = 1
    metrics = backend.train([{"prompt": "Question:\n", "completion": completion}])
    assert metrics["examples"] == 1 and metrics["target_tokens"] > 0


def test_mps_forward_training_and_generation(tmp_path):
    if not torch.backends.mps.is_available():
        pytest.skip("MPS requires an Apple Silicon runner")
    create_tiny_model(tmp_path / "base")
    config = Config(
        device="mps",
        max_seq_length=128,
        max_new_tokens=10,
        max_steps=1,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
    )
    backend = TorchBackend(str(tmp_path / "base"), config, training=True)
    try:
        metrics = backend.train([{"prompt": "Q:\nA:", "completion": " Two (a)."}])
        assert metrics["optimizer_steps"] == 1
        assert len(backend.generate(["Q:\nA:"])) == 1
        assert next(backend.model.parameters()).device.type == "mps"
    finally:
        backend.close()
