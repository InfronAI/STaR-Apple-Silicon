"""Offline integration fixture, deliberately overfit; not a reasoning benchmark."""

from dataclasses import replace
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import GPTJConfig, GPTJForCausalLM, PreTrainedTokenizerFast

from .backend import TorchBackend, set_seed
from .config import Config
from .data import Example, write_json, write_jsonl
from .pipeline import run


def create_tiny_model(directory, seed=42):
    set_seed(seed)
    special = ["<|pad|>", "<|endoftext|>", "<|unk|>"]
    alphabet = special + sorted(pre_tokenizers.ByteLevel.alphabet())
    tokenizer = Tokenizer(
        models.BPE(
            vocab={token: i for i, token in enumerate(alphabet)}, merges=[], unk_token=special[2]
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token=special[0], eos_token=special[1], unk_token=special[2]
    )
    config = GPTJConfig(
        vocab_size=len(alphabet),
        n_positions=256,
        n_embd=64,
        n_layer=2,
        n_head=2,
        rotary_dim=32,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPTJForCausalLM(config)
    model.save_pretrained(directory, safe_serialization=True)
    fast.save_pretrained(directory)


def smoke(output, device="auto", warmup_steps=200):
    root = Path(output).expanduser().resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Smoke output must be empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    raw_model, base = root / "random_model", root / "fixture_model"
    create_tiny_model(raw_model)
    choices = (("a", "two"), ("b", "three"))
    first = Example("toy-1", "commonsenseqa", "1 + 1?", "a", choices)
    second = Example("toy-2", "commonsenseqa", "1 + 2?", "b", choices)
    a = " Adding gives two (a)."
    b = " Adding gives three (b)."
    # A controlled pretraining fixture: one known error, recoverable when hinted.
    # The actual STaR loop receives only questions/answers and generates all rationales.
    fixture_records = [
        {"prompt": first.prompt(), "completion": a},
        {"prompt": first.prompt(True), "completion": a},
        {"prompt": second.prompt(), "completion": a},
        {"prompt": second.prompt(True), "completion": b},
    ]
    config = Config(
        model=str(raw_model),
        device=device,
        local_files_only=True,
        few_shot=0,
        max_seq_length=256,
        max_new_tokens=40,
        batch_size=4,
        gradient_accumulation_steps=1,
        max_steps=warmup_steps,
        gradient_checkpointing=False,
        learning_rate=0.003,
        z_loss=0.0,
    )
    print(
        "Building a deliberately overfit synthetic GPT-J fixture (not benchmark training)",
        flush=True,
    )
    backend = TorchBackend(str(raw_model), config, training=True)
    try:
        warmup = backend.train(fixture_records)
        backend.save(base)
        outputs = backend.generate([first.prompt(), second.prompt(), second.prompt(True)])
    finally:
        backend.close()
    write_json(
        root / "fixture.json",
        {
            "warning": "Synthetic, deliberately overfit integration fixture; not generalization evidence",
            "training": warmup,
            "outputs": outputs,
        },
    )
    if [o.strip() for o in outputs] != [a.strip(), a.strip(), b.strip()]:
        raise RuntimeError(
            f"Tiny fixture did not converge to its control outputs: {outputs}; increase --warmup-steps"
        )
    rows = [
        {
            "id": e.id,
            "question": {
                "stem": e.question,
                "choices": [{"label": label.upper(), "text": text} for label, text in choices],
            },
            "answerKey": e.answer.upper(),
        }
        for e in (first, second)
    ]
    data = root / "data" / "commonsenseqa"
    write_jsonl(data / "train_rand_split.jsonl", rows)
    # Deliberate overlap is confined to this fixture; the report explicitly labels it.
    write_jsonl(data / "dev_rand_split.jsonl", rows)
    experiment = replace(
        config,
        model=str(base),
        data_root=str(root / "data"),
        output_dir=str(root / "run"),
        iterations=2,
        train_samples=2,
        eval_samples=2,
        generation_batch_size=2,
        max_steps=3,
        batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
    )
    write_json(root / "config.json", experiment.to_dict())
    result = run(experiment)
    if (
        result["trained_iterations"] != 2
        or result["history"][0]["generation"]["rationalization_correct"] != 1
    ):
        raise RuntimeError(
            "Smoke run did not exercise both training iterations and rationalization"
        )
    resumed = run(experiment, resume=True)
    if resumed != result:
        raise RuntimeError("Resume changed the committed smoke result")
    report = {
        "status": "passed",
        "device": device,
        "fixture_only": True,
        "iterations": 2,
        "rationalization_checked": True,
        "resume_checked": True,
        "model": result["model"],
    }
    write_json(root / "smoke_result.json", report)
    return report
