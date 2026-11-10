#!/usr/bin/env python3
"""Run and record the real STaR workflow, then independently audit its artifacts."""

import argparse
import codecs
import errno
import fcntl
import hashlib
import json
import math
import os
import pty
import re
import shlex
import shutil
import struct
import subprocess
import sys
import termios
import time
from datetime import datetime, timezone
from pathlib import Path

from star.config import Config
from star.data import answers_equal, load_examples, parse_answer, read_jsonl, write_json

PROJECT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def hashes(folder):
    result = {}
    for path in sorted(Path(folder).rglob("*")):
        if path.is_file() and path.name != ".run.lock":
            with path.open("rb") as stream:
                result[str(path.relative_to(folder))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
    return result


def portable(value):
    if isinstance(value, dict):
        return {key: portable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable(item) for item in value]
    return value.replace(str(PROJECT) + "/", "") if isinstance(value, str) else value


class Recorder:
    """Capture actual PTY output in asciicast v2 with unmodified event timing."""

    def __init__(self, folder):
        self.folder = folder
        self.started = time.monotonic()
        self.stages = []
        self.cast = (folder / "terminal.cast").open("w", encoding="utf-8")
        self.cast.write(
            json.dumps(
                {
                    "version": 2,
                    "width": 132,
                    "height": 40,
                    "timestamp": int(time.time()),
                    "title": "STaR on Apple Silicon: recorded end-to-end experiment",
                    "env": {"TERM": "xterm-256color", "SHELL": "/bin/zsh"},
                }
            )
            + "\n"
        )

    def emit(self, text):
        # Remove the machine-specific project prefix; all program output is retained.
        text = text.replace(str(PROJECT) + "/", "")
        text = text.replace(str(PROJECT), ".")
        self.cast.write(json.dumps([round(time.monotonic() - self.started, 6), "o", text]) + "\n")
        self.cast.flush()
        sys.stdout.write(text)
        sys.stdout.flush()
        return text

    def stage(self, title):
        item = {
            "number": len(self.stages) + 1,
            "title": title,
            "start": time.monotonic() - self.started,
            "commands": [],
        }
        self.stages.append(item)
        self.emit(f"\n[STAGE {item['number']}/10] {title}\n")
        return item

    def command(self, args, name):
        self.emit("$ " + shlex.join(map(str, args)) + "\n")
        master, slave = pty.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 132, 0, 0))
        environment = dict(
            os.environ,
            PYTHONUNBUFFERED="1",
            NO_COLOR="1",
            TERM="dumb",
            TOKENIZERS_PARALLELISM="false",
        )
        before = time.monotonic()
        process = subprocess.Popen(
            args, cwd=PROJECT, stdin=subprocess.DEVNULL, stdout=slave, stderr=slave, env=environment
        )
        os.close(slave)
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        output = []
        try:
            while True:
                try:
                    chunk = os.read(master, 65536)
                except OSError as error:
                    if error.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break
                output.append(self.emit(decoder.decode(chunk)))
        finally:
            os.close(master)
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait()
        trailing = decoder.decode(b"", final=True)
        if trailing:
            output.append(self.emit(trailing))
        duration = time.monotonic() - before
        log = "".join(output).replace("\r\n", "\n")
        (self.folder / f"{name}.log").write_text(log, encoding="utf-8")
        self.stages[-1]["commands"].append(
            {
                "argv": portable(list(map(str, args))),
                "exit_code": process.returncode,
                "seconds": duration,
                "log": f"{name}.log",
            }
        )
        self.emit(f"[exit {process.returncode}; elapsed {duration:.2f}s]\n")
        if process.returncode:
            raise RuntimeError(f"Command failed: {name} (exit {process.returncode})")
        return log

    def close(self):
        elapsed = time.monotonic() - self.started
        for index, stage in enumerate(self.stages):
            end = self.stages[index + 1]["start"] if index + 1 < len(self.stages) else elapsed
            stage["seconds"] = end - stage["start"]
        self.cast.close()
        write_json(self.folder / "stages.json", self.stages)


def audit(config, root, report, baseline_files, before_resume):
    summary = read_json(root / "run/summary.json")
    assert summary["trained_iterations"] == config.iterations, "Not every iteration trained"
    assert config.max_steps is None, "The E2E experiment must use full epoch training"
    examples = {item.id: item for item in load_examples(config, "train")}
    dev = load_examples(config, "dev")
    assert not {e.question for e in examples.values()} & {e.question for e in dev}
    assert hashes(root / "run/base_model") == baseline_files, "Base snapshot was modified"
    assert hashes(root / "run") == before_resume, "Resume changed committed run content"
    assert read_json(root / "reload/metrics.json") == summary["history"][-1]["evaluation"]
    assert read_jsonl(root / "reload/predictions.jsonl") == read_jsonl(
        root / "run" / f"iteration_{config.iterations:04d}" / "evaluation.jsonl"
    )
    checks = [
        "disjoint_train_dev",
        "full_epoch_training",
        "all_iterations_trained",
        "immutable_base",
        "resume_content_unchanged",
        "reloaded_evaluation_identical",
    ]
    iterations = []
    for number, metrics in enumerate(summary["history"], 1):
        folder = root / "run" / f"iteration_{number:04d}"
        predictions, records = (
            read_jsonl(folder / "generations.jsonl"),
            read_jsonl(folder / "accepted.jsonl"),
        )
        direct = [row for row in predictions if row["source"] == "generation"]
        hinted = [row for row in predictions if row["source"] == "rationalization"]
        failed = {row["id"] for row in direct if not row["correct"]}
        assert len(direct) == len(examples) and {row["id"] for row in direct} == set(examples)
        assert {row["id"] for row in hinted} == failed, "Not all failed examples were rationalized"
        assert len({row["id"] for row in records}) == len(records)
        for row in predictions:
            prediction, _ = parse_answer(row["output"], row["task"])
            correct = answers_equal(prediction, examples[row["id"]].answer, row["task"])
            assert correct == row["correct"]
        for record in records:
            example = examples[record["id"]]
            assert record["prompt"] == example.prompt(), "Training prompt includes a hint"
            source = next(
                row
                for row in predictions
                if (row["id"], row["source"]) == (record["id"], record["source"])
            )
            assert record["completion"] == parse_answer(source["output"], record["task"])[1]
            assert source["correct"], "A rejected rationale was used for training"
        training = metrics["training"]
        assert training["dropped_overlength"] == 0
        assert training["examples"] == len(records) and training["examples_seen"] >= len(records)
        microbatches = math.ceil(
            math.ceil(len(records) / config.batch_size)
            * (config.epochs + config.add_epochs * (number - 1))
        )
        assert training["optimizer_steps"] == math.ceil(
            microbatches / config.gradient_accumulation_steps
        )
        assert metrics["training_base"] == str(root / "run/base_model")
        expected_source = (
            root / "run" / ("base_model" if number == 1 else f"iteration_{number - 1:04d}/model")
        )
        assert metrics["generation_model"] == str(expected_source)
        assert hashes(folder / "model")["model.safetensors"] != baseline_files["model.safetensors"]
        evaluation = read_jsonl(folder / "evaluation.jsonl")
        assert len(evaluation) == len(dev)
        assert all(
            row["source"] == "generation" and "(CORRECT)" not in row["generation_prompt"]
            for row in evaluation
        )
        target = report / f"iteration_{number:04d}"
        target.mkdir()
        for name in ("generations.jsonl", "accepted.jsonl", "evaluation.jsonl", "metrics.json"):
            content = (folder / name).read_text(encoding="utf-8")
            (target / name).write_text(content.replace(str(PROJECT) + "/", ""), encoding="utf-8")
        iterations.append(metrics)
    checks += [
        "no_dropped_generation_tail",
        "only_failed_questions_rationalized",
        "correctness_recomputed",
        "accepted_rationales_match_actual_generations",
        "no_hint_in_training_prompt",
        "all_accepted_records_seen",
        "expected_optimizer_steps",
        "each_round_restarts_from_base",
        "generation_uses_previous_model",
        "weights_changed",
        "unhinted_dev_evaluation",
    ]
    return {"checks": {check: "passed" for check in checks}, "iterations": iterations}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/commonsenseqa.json")
    parser.add_argument("--output-dir", default="runs/e2e-2026-11-10")
    parser.add_argument("--report-dir", default="docs/experiments/2026-11-10-e2e")
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    args = parser.parse_args()
    root, report = Path(args.output_dir).resolve(), Path(args.report_dir).resolve()
    for folder in (root, report):
        if folder.exists() and any(folder.iterdir()):
            raise SystemExit(f"Choose an empty output directory: {folder}")
        folder.mkdir(parents=True, exist_ok=True)
    config = Config.load(args.config)
    config.device, config.output_dir = args.device, str(root / "run")
    assert (
        not config.few_shot_train and config.rationalize and config.rationalization_probability == 1
    )
    config.validate()
    write_json(root / "config.json", config.to_dict())
    archive_config = config.to_dict()
    archive_config.update(
        data_root=os.path.relpath(config.data_root, report),
        output_dir=os.path.relpath(config.output_dir, report),
    )
    write_json(report / "config.json", archive_config)
    sources = {}
    for path in [
        *PROJECT.glob("src/star/*.py"),
        *PROJECT.glob("tests/*.py"),
        Path(__file__),
        PROJECT / "pyproject.toml",
        PROJECT / "requirements.lock",
    ]:
        sources[str(path.relative_to(PROJECT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(report / "source-sha256.json", sources)
    recorder = Recorder(report)
    python = sys.executable
    cli = [python, "-m", "star"]
    started = datetime.now(timezone.utc).isoformat()
    result = {"status": "running", "started_utc": started}
    try:
        recorder.emit("STaR / Apple Silicon / actual end-to-end experiment\n")
        recorder.emit(
            f"GPT-2: {config.train_samples} train / {config.eval_samples} dev / {config.iterations} rounds / full epoch schedule\n"
        )
        recorder.emit(
            "Synthetic GPT-J checks branch coverage only; benchmark metrics come from original CommonsenseQA.\n"
        )
        recorder.stage("Environment and MPS execution")
        hardware = json.loads(
            recorder.command(cli + ["doctor", "--device", args.device], "01-doctor")
        )
        recorder.stage("Tests and dependency consistency")
        tests = recorder.command([python, "-m", "pytest", "-q"], "02-tests")
        recorder.command([python, "-m", "pip", "check"], "02-dependencies")
        recorder.stage("Controlled GPT-J branch coverage (synthetic fixture)")
        recorder.command(
            cli + ["smoke", "--device", args.device, "--output-dir", str(root / "fixture")],
            "03-fixture",
        )
        recorder.stage("Untrained GPT-2 baseline on the held-out dev subset")
        common = ["--config", str(root / "config.json")]
        recorder.command(
            cli + ["evaluate", *common, "--output-dir", str(root / "baseline")], "04-baseline"
        )
        recorder.stage("Two complete STaR rounds on original CommonsenseQA")
        recorder.command(cli + ["run", *common], "05-star")
        summary = read_json(root / "run/summary.json")
        baseline_files = hashes(root / "run/base_model")
        assert baseline_files == read_json(root / "run/manifest.json")["base_files"]
        recorder.stage("Reload the final checkpoint and independently evaluate")
        recorder.command(
            cli
            + [
                "evaluate",
                *common,
                "--model",
                summary["model"],
                "--output-dir",
                str(root / "reload"),
            ],
            "06-reload",
        )
        recorder.stage("Generate from the saved checkpoint")
        prompt = load_examples(config, "dev")[0].prompt()
        generated = recorder.command(
            cli + ["generate", *common, "--model", summary["model"], "--prompt", prompt],
            "07-generate",
        )
        write_json(
            report / "saved-model-generation.json", {"prompt": prompt, **json.loads(generated)}
        )
        recorder.stage("Resume without retraining or changing committed artifacts")
        before_resume = hashes(root / "run")
        recorder.command(cli + ["run", *common, "--resume"], "08-resume")
        recorder.stage("Independent artifact audit")
        audited = audit(config, root, report, baseline_files, before_resume)
        for check in audited["checks"]:
            recorder.emit(f"PASS {check}\n")
        result = {
            "status": "passed",
            "started_utc": started,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "hardware": hardware,
            "config": portable(config.to_dict()),
            "tests_passed": int(re.search(r"(\d+) passed", tests).group(1)),
            "baseline": read_json(root / "baseline/metrics.json"),
            "fixture": portable(read_json(root / "fixture/smoke_result.json")),
            "audit": portable(audited),
            "final_model": portable(summary["model"]),
        }
        for source, name in [
            (root / "baseline/predictions.jsonl", "baseline-predictions.jsonl"),
            (root / "reload/predictions.jsonl", "reloaded-predictions.jsonl"),
        ]:
            shutil.copyfile(source, report / name)
        write_json(report / "result.json", result)
        recorder.stage("Results and scope")
        recorder.emit(
            f"RESULT: PASS | {result['tests_passed']} tests | {len(audited['checks'])} independent artifact checks\n"
        )
        recorder.emit(
            f"Baseline dev: {result['baseline']['correct']}/{result['baseline']['total']}\n"
        )
        for item in audited["iterations"]:
            g, t, e = item["generation"], item["training"], item["evaluation"]
            recorder.emit(
                f"Round {item['iteration']}: generated {g['correct']}/{g['total']}; recovered {g['rationalization_correct']}; "
                f"accepted {g['accepted']}; optimizer steps {t['optimizer_steps']}; dev {e['correct']}/{e['total']}\n"
            )
        recorder.emit(
            "This is a bounded engineering experiment, not a reproduction of paper-scale accuracy.\n"
        )
        recorder.emit(f"Report artifacts: {report}\nFinal checkpoint: {summary['model']}\n")
    except Exception as error:
        result.update(status="failed", error=str(error))
        write_json(report / "result.json", result)
        recorder.emit(f"\nFAILED: {error}\n")
        raise
    finally:
        recorder.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
