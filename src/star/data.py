"""Original task formats, answer checking and hint-free training records."""

import json
import random
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: invalid JSON: {error.msg}") from error
    return rows


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class Example:
    id: str
    task: str
    question: str
    answer: str
    choices: tuple[tuple[str, str], ...] = ()

    def prompt(self, hint=False):
        if self.task == "commonsenseqa":
            context = f"Q: {self.question}\nAnswer Choices:\n"
            for label, text in self.choices:
                mark = " (CORRECT)" if hint and label.lower() == self.answer.lower() else ""
                context += f"({label.lower()}) {text}{mark}\n"
            return context + "A:"
        if self.task == "gsm":
            suffix = f" ({self.answer})" if hint else ""
            return f"Q: {self.question}{suffix}\nA:"
        context = f"Input:\n{self.question}\nTarget:"
        return context + (f" {self.answer}" if hint else "") + "\n"


def load_examples(config, split):
    override = config.train_file if split == "train" else config.eval_file
    path = (
        Path(override)
        if override
        else Path(config.data_root) / config.task / f"{split}_rand_split.jsonl"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found: {path}. For arithmetic run star prepare-arithmetic first."
        )
    examples = []
    for i, row in enumerate(read_jsonl(path)):
        try:
            if config.task == "commonsenseqa":
                choices = tuple((c["label"], c["text"]) for c in row["question"]["choices"])
                answer = row["answerKey"].strip().lower()
                if answer not in {label.lower() for label, _ in choices}:
                    raise ValueError("answerKey is not one of the choices")
                example = Example(
                    str(row.get("id", i)), config.task, row["question"]["stem"], answer, choices
                )
            else:
                answer = row["answer"].rsplit("####", 1)[-1].strip()
                if not answer:
                    raise ValueError("empty answer")
                example = Example(str(row.get("id", i)), config.task, row["question"], answer)
            examples.append(example)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{path}: example {i + 1} is not a labeled {config.task} example: {error}"
            ) from error
    if len({e.id for e in examples}) != len(examples):
        raise ValueError(f"Duplicate example IDs in {path}")
    if not examples:
        raise ValueError(f"Empty dataset: {path}")
    if split == "train":
        random.Random(config.seed).shuffle(examples)
        return examples[: config.train_samples]
    return examples[: config.eval_samples]


def load_demonstrations(config, hint=False):
    if not config.few_shot:
        return []
    suffix = "_answer_key" if hint else ""
    direct = "_direct" if config.direct else ""
    path = Path(config.data_root) / config.task / f"prompts{direct}{suffix}.txt"
    if not path.exists() and config.direct:
        # The original GSM release has no prompts_direct.txt.
        path = Path(config.data_root) / config.task / f"prompts{suffix}.txt"
    if not path.is_file():
        raise FileNotFoundError(
            f"Few-shot prompts not found: {path}; set few_shot=0 or prepare the task data."
        )
    prompts = [p.strip() for p in path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    if config.direct:
        direct_prompts = []
        separator = "\nTarget:" if config.task == "arithmetic" else "\nA:"
        for prompt in prompts:
            question, completion = prompt.split(separator, 1)
            prediction, _ = parse_answer(completion, config.task)
            if prediction is None:
                raise ValueError(f"Cannot extract demonstration answer in {path}")
            answer = (
                f" ({prediction})."
                if config.task == "commonsenseqa"
                else (f" #### {prediction}" if config.task == "gsm" else f"\n{prediction}")
            )
            direct_prompts.append(question + separator + answer)
        prompts = direct_prompts
    return prompts[: config.few_shot]


def parse_answer(completion, task):
    """Return (prediction, usable completion), never reading a later generated question."""
    text = completion.split("<|endoftext|>", 1)[0]
    text = re.split(r"\nQ:|\nInput:", text, maxsplit=1)[0].rstrip()
    if task == "commonsenseqa":
        text = re.split(r"\n\s*\n", text, maxsplit=1)[0].rstrip()
        match = re.search(r"\(([a-eA-E])\)[.!]?\s*$", text)
        if not match:
            match = re.fullmatch(r"\s*([a-eA-E])[.!]?", text)
        return (match.group(1).lower(), text) if match else (None, text)
    if task == "gsm":
        match = re.search(r"####\s*([^\n]+)", text)
        return (match.group(1).strip(), text[: match.end()]) if match else (None, text)
    if "<scratch>" in text and "</scratch>" not in text:
        return None, text
    tail = text.split("</scratch>", 1)[-1].strip()
    line = tail.split("\n", 1)[0]
    if re.fullmatch(r"\d(?:[ \t]*\d)*", line):
        end = text.find("</scratch>") + len("</scratch>") if "</scratch>" in text else 0
        end += len(text[end:]) - len(text[end:].lstrip()) + len(line)
        return line, text[:end]
    return None, text


def answers_equal(prediction, target, task):
    if prediction is None:
        return False
    if task == "commonsenseqa":
        return prediction.strip().lower() == target.strip().lower()
    if task == "arithmetic":
        return prediction.replace(" ", "").replace("\t", "").lstrip("0") == target.replace(
            " ", ""
        ).lstrip("0")
    try:
        left = Decimal(prediction.replace(",", "").strip())
        right = Decimal(target.replace(",", "").strip())
        return left.is_finite() and right.is_finite() and left == right
    except InvalidOperation:
        return prediction.strip().lower() == target.strip().lower()


def arithmetic_completion(first, second):
    def spaced(number):
        return " ".join(str(number))

    width = max(len(str(first)), len(str(second)))
    carry, running = 0, ""
    lines = ["<scratch>", f"{spaced(first)} + {spaced(second)} , C: 0"]
    for a, b in reversed(list(zip(str(first).zfill(width), str(second).zfill(width)))):
        total = int(a) + int(b) + carry
        lines.append(f"{a} + {b} , {running}C: {carry}")
        carry = total // 10
        running = f"{total % 10} {running}"
    lines.extend(
        [f", {running}C: {carry}", spaced(first + second), "</scratch>", spaced(first + second)]
    )
    return "\n".join(lines)


def prepare_arithmetic(root, train_count=200, eval_count=50, max_digits=3, seed=42):
    if min(train_count, eval_count, max_digits) < 1 or max_digits > 8:
        raise ValueError("Counts must be positive and max_digits must be between 1 and 8")
    capacity = sum((9 * 10 ** (digits - 1)) ** 2 for digits in range(1, max_digits + 1))
    if train_count + eval_count + 3 > capacity:
        raise ValueError("Too many unique examples for max_digits; increase the digit range")
    directory = Path(root) / "arithmetic"
    names = [
        "train_rand_split.jsonl",
        "dev_rand_split.jsonl",
        "prompts.txt",
        "prompts_answer_key.txt",
    ]
    if any((directory / name).exists() for name in names):
        raise FileExistsError(
            f"Arithmetic data already exists in {directory}; choose a new --data-root"
        )
    rng, seen, rows = random.Random(seed), set(), []
    while len(rows) < train_count + eval_count + 3:
        digits = rng.randint(1, max_digits)
        first, second = [rng.randrange(10 ** (digits - 1), 10**digits) for _ in range(2)]
        if (first, second) in seen:
            continue
        seen.add((first, second))
        rows.append(
            {
                "id": f"{first}+{second}",
                "question": f"{' '.join(str(first))} + {' '.join(str(second))}",
                "answer": " ".join(str(first + second)),
                "rationale": arithmetic_completion(first, second),
            }
        )
    directory.mkdir(parents=True, exist_ok=True)
    write_jsonl(directory / names[0], rows[3 : 3 + train_count])
    write_jsonl(directory / names[1], rows[3 + train_count :])
    for hint, name in ((False, names[2]), (True, names[3])):
        prompts = []
        for row in rows[:3]:
            example = Example(row["id"], "arithmetic", row["question"], row["answer"])
            prompts.append(example.prompt(hint) + row["rationale"])
        (directory / name).write_text("\n\n".join(prompts) + "\n", encoding="utf-8")
    return {
        "directory": str(directory.resolve()),
        "train": train_count,
        "dev": eval_count,
        "seed": seed,
    }
