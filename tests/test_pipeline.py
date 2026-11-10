from dataclasses import replace
from pathlib import Path

import pytest

from star.config import Config
from star.data import Example, write_json, write_jsonl
from star.pipeline import collect, run


class ControlledBackend:
    calls = []
    resolved_revision = "test-fixture"

    def __init__(self, source, config, training=False):
        self.source = str(source)

    def fit_prompt(self, question, demos):
        return "\n\n".join([*demos, question])

    def generate(self, prompts):
        return [
            " A generated rationale (b)."
            if "recover" in p and "CORRECT" in p
            else " A generated rationale (a)."
            for p in prompts
        ]

    def save(self, path):
        write_json(Path(path) / "fixture.json", {"source": self.source})

    def train(self, records, iteration):
        self.calls.append((self.source, iteration, records))
        return {"optimizer_steps": 1}

    def close(self):
        pass


def examples():
    choices = (("a", "first"), ("b", "second"))
    return [
        Example(str(i), "commonsenseqa", question, answer, choices)
        for i, (question, answer) in enumerate(
            [("ok", "a"), ("recover", "b"), ("wrong", "b"), ("tail", "a")]
        )
    ]


def test_filter_rationalize_remove_hint_and_retain_tail():
    config = Config(few_shot=0, generation_batch_size=3)
    accepted, predictions, stats = collect(
        ControlledBackend("base", config), examples(), config, [], [], True
    )
    assert stats == {
        "total": 4,
        "correct": 2,
        "rationalization_attempted": 2,
        "rationalization_correct": 1,
        "accuracy": 0.5,
        "accepted": 3,
    }
    assert len(predictions) == 6
    assert all("CORRECT" not in r["prompt"] for r in accepted)
    assert {r["id"] for r in accepted} == {"0", "1", "3"}
    _, predictions, stats = collect(
        ControlledBackend("base", config), examples(), config, [], [], False
    )
    assert len(predictions) == 4 and stats["rationalization_attempted"] == 0
    assert all("CORRECT" not in row["generation_prompt"] for row in predictions)


def config_with_data(tmp_path, wrong_only=False):
    rows = [
        {
            "id": e.id,
            "question": {
                "stem": e.question,
                "choices": [{"label": label, "text": text} for label, text in e.choices],
            },
            "answerKey": e.answer,
        }
        for e in (examples()[2:3] if wrong_only else examples())
    ]
    for split in ("train", "dev"):
        write_jsonl(tmp_path / "commonsenseqa" / f"{split}_rand_split.jsonl", rows)
    return Config(
        model="fixture",
        data_root=str(tmp_path),
        output_dir=str(tmp_path / "run"),
        few_shot=0,
        iterations=2,
    )


def test_iteration_resets_to_base_and_resume_is_idempotent(tmp_path):
    config = config_with_data(tmp_path)
    ControlledBackend.calls = []
    result = run(config, backend_factory=ControlledBackend)
    base = str(tmp_path / "run/base_model")
    assert [c[0] for c in ControlledBackend.calls] == [base, base]
    assert result["history"][1]["generation_model"].endswith("iteration_0001/model")
    assert result["trained_iterations"] == 2
    assert result == run(config, resume=True, backend_factory=ControlledBackend)
    assert len(ControlledBackend.calls) == 2
    with pytest.raises(ValueError, match="cannot reduce"):
        run(replace(config, iterations=1), resume=True, backend_factory=ControlledBackend)
    with pytest.raises(ValueError, match="Resume config/data"):
        run(replace(config, seed=99), resume=True, backend_factory=ControlledBackend)
    with pytest.raises(FileExistsError):
        run(config, backend_factory=ControlledBackend)
    # A modified checkpoint cannot silently become a new base.
    write_json(Path(base) / "fixture.json", {"changed": True})
    with pytest.raises(ValueError, match="Base model snapshot"):
        run(config, resume=True, backend_factory=ControlledBackend)


def test_no_success_never_trains_on_reference_answers(tmp_path):
    config = config_with_data(tmp_path, wrong_only=True)
    ControlledBackend.calls = []
    result = run(config, backend_factory=ControlledBackend)
    assert not ControlledBackend.calls
    assert result["trained_iterations"] == 0
    assert result["model"].endswith("base_model")
    assert all(r["status"] == "skipped_no_correct_examples" for r in result["history"])


def test_interrupted_iteration_is_replayed(tmp_path):
    config = config_with_data(tmp_path)

    class Interrupted(ControlledBackend):
        def train(self, records, iteration):
            raise RuntimeError("interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        run(config, backend_factory=Interrupted)
    assert not (tmp_path / "run/iteration_0001/metrics.json").exists()
    result = run(config, resume=True, backend_factory=ControlledBackend)
    assert result["trained_iterations"] == 2
