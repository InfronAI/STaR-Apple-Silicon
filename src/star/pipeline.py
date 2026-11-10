"""STaR: generate with theta(t-1), filter, rationalize, retrain theta(0), evaluate."""

import fcntl
import hashlib
import json
import random
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .backend import TorchBackend, set_seed
from .data import (
    answers_equal,
    load_demonstrations,
    load_examples,
    parse_answer,
    write_json,
    write_jsonl,
)


def collect(
    backend, examples, config, demonstrations, hint_demonstrations, rationalize=False, seed=None
):
    accepted, predictions, wrong = [], [], []
    rng = random.Random(config.seed if seed is None else seed)
    stats = {
        "total": len(examples),
        "correct": 0,
        "rationalization_attempted": 0,
        "rationalization_correct": 0,
    }

    def generate_group(group, hint):
        demos = hint_demonstrations if hint else demonstrations
        prompts = [backend.fit_prompt(example.prompt(hint), demos) for example in group]
        outputs = backend.generate(prompts)
        if len(outputs) != len(group):
            raise RuntimeError("Backend returned a different number of completions than prompts")
        failed = []
        for example, prompt, output in zip(group, prompts, outputs):
            prediction, completion = parse_answer(output, example.task)
            correct = answers_equal(prediction, example.answer, example.task)
            source = "rationalization" if hint else "generation"
            predictions.append(
                {
                    "id": example.id,
                    "task": example.task,
                    "source": source,
                    "generation_prompt": prompt,
                    "output": output,
                    "prediction": prediction,
                    "answer": example.answer,
                    "correct": correct,
                }
            )
            if correct:
                stats["rationalization_correct" if hint else "correct"] += 1
                # Only the model's completion is accepted. Never substitute a reference rationale.
                training_prompt = backend.fit_prompt(
                    example.prompt(), demonstrations if config.few_shot_train else []
                )
                accepted.append(
                    {
                        "id": example.id,
                        "task": example.task,
                        "prompt": training_prompt,
                        "completion": completion,
                        "answer": example.answer,
                        "source": source,
                    }
                )
            elif not hint:
                failed.append(example)
        return failed

    size = config.generation_batch_size
    for start in range(0, len(examples), size):
        wrong.extend(generate_group(examples[start : start + size], False))
        if start == 0 or (start + size) % 10 == 0 or start + size >= len(examples):
            print(
                f"  generate {min(start + size, len(examples))}/{len(examples)}: correct={stats['correct']}",
                flush=True,
            )
    if rationalize:
        candidates = [
            example for example in wrong if rng.random() < config.rationalization_probability
        ]
        stats["rationalization_attempted"] = len(candidates)
        for start in range(0, len(candidates), size):
            generate_group(candidates[start : start + size], True)
            if start == 0 or (start + size) % 10 == 0 or start + size >= len(candidates):
                print(
                    f"  rationalize {min(start + size, len(candidates))}/{len(candidates)}: recovered={stats['rationalization_correct']}",
                    flush=True,
                )
    stats["accuracy"] = stats["correct"] / len(examples) if examples else 0.0
    stats["accepted"] = len(accepted)
    return accepted, predictions, stats


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def checkpoint_hashes(path):
    result = {}
    for file in sorted(Path(path).iterdir()):
        if file.is_file():
            with file.open("rb") as stream:
                result[file.name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


@contextmanager
def run_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another process is using {root}") from error
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def run(config, resume=False, backend_factory=TorchBackend):
    config.validate()
    train = load_examples(config, "train")
    evaluation = load_examples(config, "dev")
    demonstrations = load_demonstrations(config)
    hint_demonstrations = load_demonstrations(config, hint=True) if config.rationalize else []
    fingerprint = digest(
        {
            "train": [asdict(e) for e in train],
            "dev": [asdict(e) for e in evaluation],
            "prompts": demonstrations,
            "hint_prompts": hint_demonstrations,
        }
    )
    root = Path(config.output_dir).resolve()
    base = root / "base_model"
    with run_lock(root):
        manifest_path = root / "manifest.json"
        settings = config.to_dict()
        settings["output_dir"] = str(root)
        if manifest_path.exists():
            if not resume:
                raise FileExistsError(
                    f"Run already exists in {root}; use --resume or a new output directory"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            completed = [
                int(path.parent.name.split("_")[-1])
                for path in root.glob("iteration_*/metrics.json")
            ]
            if config.iterations < max([manifest["config"]["iterations"], *completed]):
                raise ValueError("Resume may increase iterations but cannot reduce the run history")
            previous = dict(manifest["config"])
            current = dict(settings)
            previous.pop("iterations")
            current.pop("iterations")
            if previous != current or manifest["data_sha256"] != fingerprint:
                raise ValueError(
                    "Resume config/data differs from the original run; only iterations may change"
                )
            if checkpoint_hashes(base) != manifest["base_files"]:
                raise ValueError(
                    "Base model snapshot changed or is incomplete; use a new output directory"
                )
        else:
            if resume:
                raise FileNotFoundError(f"No committed run to resume in {root}")
            if any(p.name != ".run.lock" for p in root.iterdir()):
                raise FileExistsError(f"Output directory is not empty: {root}")
            print(f"Saving immutable base model: {config.model}", flush=True)
            set_seed(config.seed)
            backend = backend_factory(config.model, config, training=True)
            try:
                backend.save(base)
                manifest = {
                    "format": 1,
                    "config": settings,
                    "data_sha256": fingerprint,
                    "model_revision": backend.resolved_revision,
                    "base_files": checkpoint_hashes(base),
                }
            finally:
                backend.close()
            write_json(manifest_path, manifest)

        current_model = base
        history = []
        for iteration in range(1, config.iterations + 1):
            folder = root / f"iteration_{iteration:04d}"
            result_path = folder / "metrics.json"
            if result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if result["status"] == "trained":
                    current_model = folder / "model"
                    if checkpoint_hashes(current_model) != result["model_files"]:
                        raise ValueError(f"Checkpoint changed or is incomplete: {current_model}")
                history.append(result)
                continue
            print(
                f"Iteration {iteration}/{config.iterations}: generating from {current_model}",
                flush=True,
            )
            folder.mkdir(exist_ok=True)
            set_seed(config.seed + iteration)
            backend = backend_factory(str(current_model), config)
            try:
                records, predictions, generation = collect(
                    backend,
                    train,
                    config,
                    demonstrations,
                    hint_demonstrations,
                    rationalize=config.rationalize,
                    seed=config.seed + iteration,
                )
            finally:
                backend.close()
            write_jsonl(folder / "generations.jsonl", predictions)
            write_jsonl(folder / "accepted.jsonl", records)
            result = {
                "iteration": iteration,
                "generation_model": str(current_model),
                "training_base": str(base),
                "generation": generation,
            }
            if records:
                print(f"  retraining base model on {len(records)} accepted examples", flush=True)
                set_seed(config.seed + iteration)
                backend = backend_factory(str(base), config, training=True)
                try:
                    result["training"] = backend.train(records, iteration)
                    backend.save(folder / "model")
                finally:
                    backend.close()
                current_model = folder / "model"
                result["status"] = "trained"
                result["model_files"] = checkpoint_hashes(current_model)
            else:
                print(
                    "  no correct completions: training skipped; retaining previous model",
                    flush=True,
                )
                result["status"] = "skipped_no_correct_examples"
            set_seed(config.seed)
            backend = backend_factory(str(current_model), config)
            try:
                _, predictions, result["evaluation"] = collect(
                    backend, evaluation, config, demonstrations, [], rationalize=False
                )
            finally:
                backend.close()
            write_jsonl(folder / "evaluation.jsonl", predictions)
            # Commit only after generation, training, checkpointing and unhinted evaluation succeed.
            write_json(result_path, result)
            history.append(result)
        summary = {
            "iterations": len(history),
            "trained_iterations": sum(r["status"] == "trained" for r in history),
            "model": str(current_model),
            "history": history,
        }
        write_json(root / "summary.json", summary)
        return summary
