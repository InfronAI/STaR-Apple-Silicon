from dataclasses import replace
from pathlib import Path

import pytest

from star.config import Config
from star.data import (
    Example,
    answers_equal,
    arithmetic_completion,
    load_demonstrations,
    load_examples,
    parse_answer,
    prepare_arithmetic,
    read_jsonl,
    write_json,
)


@pytest.mark.parametrize(
    "task,text,expected",
    [
        ("commonsenseqa", " Because it absorbs ink (E).\n\nQ: next (a).", "e"),
        ("commonsenseqa", " Maybe (a), but I don't know", None),
        ("commonsenseqa", "B", "b"),
        ("gsm", " First add 2.\n#### 1,200\nQ: next", "1,200"),
        ("gsm", " It is 12", None),
        ("gsm", " First add 2.\n\nThen add 3.\n#### 5", "5"),
        ("gsm", " wrong\n\nQ: next\n#### 42", None),
        ("arithmetic", "<scratch>\n7 + 9\n</scratch>\n1 6", "1 6"),
        ("arithmetic", "<scratch>\n1 6", None),
        ("arithmetic", "\n1 6\nnoise", "1 6"),
    ],
)
def test_answer_parsing_stops_at_current_example(task, text, expected):
    assert parse_answer(text, task)[0] == expected


def test_numeric_normalization():
    assert answers_equal("1,200.00", "1200", "gsm")
    assert answers_equal("0 1 6", "16", "arithmetic")
    assert not answers_equal("NaN", "NaN", "gsm")
    assert not answers_equal(None, "a", "commonsenseqa")
    assert not answers_equal("12 dollars", "12", "gsm")


def test_hint_contains_only_final_answer():
    example = Example("one", "gsm", "How many?", "42")
    assert example.prompt() == "Q: How many?\nA:"
    assert example.prompt(True) == "Q: How many? (42)\nA:"


def test_original_data_and_missing_direct_gsm_prompts():
    root = Path(__file__).resolve().parents[1] / "datasets"
    config = Config(data_root=str(root), train_samples=3, task="gsm", direct=True)
    rows = load_examples(config, "train")
    assert len(rows) == 3
    assert "<<" not in rows[0].prompt(True)
    prompts = load_demonstrations(config)
    assert len(prompts) == 3 and all("####" in p and "<<" not in p for p in prompts)
    assert len(load_examples(replace(config, task="commonsenseqa"), "dev")) == config.eval_samples


def test_arithmetic_splits_are_unique_disjoint_reproducible(tmp_path):
    prepare_arithmetic(tmp_path / "a", 20, 10, 2, 7)
    prepare_arithmetic(tmp_path / "b", 20, 10, 2, 7)
    train = read_jsonl(tmp_path / "a/arithmetic/train_rand_split.jsonl")
    dev = read_jsonl(tmp_path / "a/arithmetic/dev_rand_split.jsonl")
    assert not {r["id"] for r in train} & {r["id"] for r in dev}
    assert train == read_jsonl(tmp_path / "b/arithmetic/train_rand_split.jsonl")
    assert parse_answer(arithmetic_completion(99, 11), "arithmetic")[0] == "1 1 0"
    with pytest.raises(FileExistsError):
        prepare_arithmetic(tmp_path / "a")
    with pytest.raises(ValueError, match="Too many"):
        prepare_arithmetic(tmp_path / "c", 100, 100, 1)


def test_config_paths_and_typos(tmp_path):
    path = tmp_path / "config.json"
    write_json(path, {"data_root": "data", "model": "./model", "output_dir": "out"})
    config = Config.load(path)
    assert config.data_root == str(tmp_path / "data")
    assert config.model == str(tmp_path / "model")
    write_json(path, {"unknwon": 1})
    with pytest.raises(ValueError, match="Unknown"):
        Config.load(path)


@pytest.mark.parametrize(
    "values",
    [
        {"iterations": 0},
        {"top_p": 2},
        {"max_steps": 0},
        {"gradient_accumulation_steps": 0},
        {"max_new_tokens": 512},
        {"learning_rate": float("nan")},
        {"rationalize": "false"},
    ],
)
def test_invalid_config(values):
    with pytest.raises(ValueError):
        Config(**values).validate()
