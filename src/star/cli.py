"""One local entry point; legacy TPU scripts are not imported."""

import argparse
import json
import sys
from pathlib import Path

from .config import Config


def parser():
    root = argparse.ArgumentParser(prog="star", description="STaR on Apple Silicon / CPU")
    sub = root.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check Python, MPS and real forward/backward execution")
    doctor.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    smoke = sub.add_parser(
        "smoke", help="Offline synthetic GPT-J integration test (not a benchmark)"
    )
    smoke.add_argument("--output-dir", default="runs/smoke")
    smoke.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    smoke.add_argument("--warmup-steps", type=int, default=200)
    prepare = sub.add_parser(
        "prepare-arithmetic", help="Generate deterministic, disjoint arithmetic splits"
    )
    prepare.add_argument("--data-root", default="datasets")
    prepare.add_argument("--train-count", type=int, default=200)
    prepare.add_argument("--eval-count", type=int, default=50)
    prepare.add_argument("--max-digits", type=int, default=3)
    prepare.add_argument("--seed", type=int, default=42)
    for name, help_text in (
        ("run", "Run the full STaR loop"),
        ("evaluate", "Evaluate without hints"),
        ("generate", "Generate a raw completion"),
        ("train", "Fine-tune on accepted JSONL"),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", required=True)
        command.add_argument("--device", choices=["auto", "mps", "cpu"])
        command.add_argument("--model", help="Override model ID or local checkpoint")
        command.add_argument("--revision", help="Hugging Face model revision (commit/tag/branch)")
        command.add_argument("--local-files-only", action="store_true")
        if name == "run":
            command.add_argument("--resume", action="store_true")
            command.add_argument("--output-dir")
            command.add_argument("--iterations", type=int)
        elif name == "generate":
            command.add_argument("--prompt", required=True)
        elif name == "train":
            command.add_argument("--records", required=True)
            command.add_argument("--output-dir", required=True)
        elif name == "evaluate":
            command.add_argument("--output-dir", required=True)
    return root


def execute(args):
    from .backend import TorchBackend, doctor, set_seed
    from .data import (
        answers_equal,
        load_demonstrations,
        load_examples,
        parse_answer,
        prepare_arithmetic,
        read_jsonl,
        write_json,
        write_jsonl,
    )
    from .pipeline import collect, run

    if args.command == "doctor":
        return doctor(args.device)
    if args.command == "prepare-arithmetic":
        return prepare_arithmetic(
            args.data_root, args.train_count, args.eval_count, args.max_digits, args.seed
        )
    if args.command == "smoke":
        if args.warmup_steps < 1:
            raise ValueError("warmup-steps must be positive")
        from .smoke import smoke

        return smoke(args.output_dir, args.device, args.warmup_steps)
    config = Config.load(args.config)
    if args.device:
        config.device = args.device
    if args.model:
        config.model = str(Path(args.model).resolve()) if Path(args.model).is_dir() else args.model
        config.revision = "main"
    if args.revision:
        config.revision = args.revision
    if args.local_files_only:
        config.local_files_only = True
    if args.command == "run":
        if args.output_dir:
            config.output_dir = str(Path(args.output_dir).expanduser().resolve())
        if args.iterations is not None:
            config.iterations = args.iterations
        result = run(config, resume=args.resume)
        return {key: value for key, value in result.items() if key != "history"}
    config.validate()
    set_seed(config.seed)
    if args.command == "generate":
        backend = TorchBackend(config.model, config)
        try:
            return {"completion": backend.generate([args.prompt])[0]}
        finally:
            backend.close()
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    if args.command == "evaluate":
        examples, prompts = load_examples(config, "dev"), load_demonstrations(config)
        backend = TorchBackend(config.model, config)
        try:
            _, predictions, metrics = collect(
                backend, examples, config, prompts, [], rationalize=False
            )
        finally:
            backend.close()
        write_jsonl(output / "predictions.jsonl", predictions)
        write_json(output / "metrics.json", metrics)
        return metrics
    records = read_jsonl(args.records)
    if not records:
        raise ValueError("Training records are empty")
    for record in records:
        prediction, _ = parse_answer(record["completion"], record["task"])
        if not answers_equal(prediction, record["answer"], record["task"]):
            raise ValueError(
                f"Training record {record.get('id')} does not have a correct final answer"
            )
    backend = TorchBackend(config.model, config, training=True)
    try:
        metrics = backend.train(records)
        backend.save(output / "model")
    finally:
        backend.close()
    write_json(output / "metrics.json", metrics)
    return metrics


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = execute(args)
    except (ValueError, OSError, RuntimeError, KeyError) as error:
        print(f"star: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
