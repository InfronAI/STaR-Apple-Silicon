"""Native PyTorch implementation: float32 MPS/CPU, full fine-tuning, local checkpoints."""

import gc
import math
import platform
import random
import subprocess
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch._dynamo  # Initialize lazy compiler imports before entering a meta-device context.
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def select_device(requested):
    available = torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise RuntimeError(
            "MPS is unavailable. Use native arm64 Python on macOS 14+ or --device cpu."
        )
    return torch.device(
        "mps" if requested == "mps" or (requested == "auto" and available) else "cpu"
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


@lru_cache(maxsize=1)
def physical_memory():
    if platform.system() == "Darwin":
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    return None


def encode_record(tokenizer, record, max_length, loss_mask="completion"):
    prompt, completion = record["prompt"], record["completion"]
    if not prompt or not completion.strip():
        raise ValueError("Training records require a nonempty prompt and completion")
    # Tokenize once: independently tokenizing prompt/answer can change BPE at the boundary.
    encoded = tokenizer(prompt + completion, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"] + [tokenizer.eos_token_id]
    if len(ids) > max_length:
        return None  # Never train on a truncated rationale whose answer has been removed.
    labels = [
        token if loss_mask == "all" or end > len(prompt) else -100
        for token, (_, end) in zip(encoded["input_ids"], encoded["offset_mapping"])
    ]
    labels.append(tokenizer.eos_token_id)
    if not any(token != -100 for token in labels[1:]):
        raise ValueError("Training record has no predictable completion tokens")
    return {"input_ids": ids, "labels": labels}


def collate(rows, pad_id, device):
    length = max(len(row["input_ids"]) for row in rows)
    ids, masks, labels = [], [], []
    for row in rows:
        missing = length - len(row["input_ids"])
        ids.append(row["input_ids"] + [pad_id] * missing)
        masks.append([1] * len(row["input_ids"]) + [0] * missing)
        labels.append(row["labels"] + [-100] * missing)
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long, device=device),
        "attention_mask": torch.tensor(masks, dtype=torch.long, device=device),
        "labels": torch.tensor(labels, dtype=torch.long, device=device),
    }


def loss_sum(logits, labels, z_loss):
    # Match the original causal shift. Padding and prompt tokens never enter the objective.
    logits, targets = logits[:, :-1].float(), labels[:, 1:]
    selected = targets != -100
    active_logits, active_targets = logits[selected], targets[selected]
    loss = F.cross_entropy(active_logits, active_targets, reduction="sum")
    if z_loss:
        # The released JAX Projection uses max-shifted logits for its z-loss.
        shifted = active_logits - active_logits.max(dim=-1, keepdim=True).values.detach()
        loss = loss + z_loss * torch.logsumexp(shifted, dim=-1).square().sum()
    return loss


class TorchBackend:
    def __init__(self, source, config, training=False):
        self.config = config
        self.device = select_device(config.device)
        self.source = str(source)
        options = {"local_files_only": config.local_files_only, "trust_remote_code": False}
        if not Path(source).is_dir():
            options["revision"] = config.revision
        model_config = AutoConfig.from_pretrained(source, **options)
        self.resolved_revision = getattr(model_config, "_commit_hash", None)
        # Fail before downloading multi-GB weights that cannot fit this Mac.
        with torch.device("meta"):
            skeleton = AutoModelForCausalLM.from_config(model_config, attn_implementation="eager")
        self.parameter_count = sum(p.numel() for p in skeleton.parameters())
        del skeleton
        memory = physical_memory()
        estimate = self.parameter_count * (16 if training else 4)
        if memory and estimate > memory * 0.8:
            raise ValueError(
                f"Model needs at least {estimate / 2**30:.1f} GiB for {'full Adam training' if training else 'float32 weights'} "
                f"before activations; this Mac has {memory / 2**30:.1f} GiB. Use a smaller model."
            )
        if self.resolved_revision and not Path(source).is_dir():
            options["revision"] = self.resolved_revision
        self.tokenizer = AutoTokenizer.from_pretrained(source, use_fast=True, **options)
        if not self.tokenizer.is_fast or self.tokenizer.eos_token_id is None:
            raise ValueError("A fast tokenizer with an EOS token is required")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=torch.float32,
            attn_implementation="eager",
            use_safetensors=True,
            **options,
        ).to(self.device)
        context_limit = getattr(self.model.config, "max_position_embeddings", config.max_seq_length)
        if config.max_seq_length > context_limit:
            raise ValueError(
                f"max_seq_length={config.max_seq_length} exceeds model context {context_limit}"
            )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.eval()

    def fit_prompt(self, question, demonstrations):
        budget = self.config.max_seq_length - self.config.max_new_tokens
        if len(self.tokenizer.encode(question, add_special_tokens=False)) > budget:
            raise ValueError(
                "Question alone exceeds the prompt budget; increase max_seq_length or reduce max_new_tokens"
            )
        demos = list(demonstrations)
        while demos:
            prompt = "\n\n".join(demos) + "\n\n" + question
            if len(self.tokenizer.encode(prompt, add_special_tokens=False)) <= budget:
                return prompt
            demos.pop(
                0
            )  # Remove whole exemplars, never truncate the actual question/answer choices.
        return question

    def generate(self, prompts):
        self.model.eval()
        results = []
        for start in range(0, len(prompts), self.config.generation_batch_size):
            batch = prompts[start : start + self.config.generation_batch_size]
            tokens = self.tokenizer(
                batch,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            width = tokens["input_ids"].shape[1]
            if width == 0 or width + self.config.max_new_tokens > self.config.max_seq_length:
                raise ValueError("Prompt is empty or exceeds the generation context budget")
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            options = {
                "do_sample": self.config.temperature > 0,
                "max_new_tokens": self.config.max_new_tokens,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True,
            }
            if self.config.temperature > 0:
                options.update(
                    temperature=self.config.temperature, top_p=self.config.top_p, top_k=0
                )
            with torch.inference_mode():
                outputs = self.model.generate(**tokens, **options)
            results.extend(
                self.tokenizer.batch_decode(outputs[:, width:].cpu(), skip_special_tokens=True)
            )
        return results

    def train(self, records, iteration=1):
        config = self.config
        rows = [
            encode_record(self.tokenizer, r, config.max_seq_length, config.loss_mask)
            for r in records
        ]
        dropped = sum(row is None for row in rows)
        rows = [row for row in rows if row is not None]
        if not rows:
            raise ValueError(
                "No complete training sequences fit max_seq_length; increase it or reduce few_shot_train"
            )
        epochs = config.epochs + config.add_epochs * (iteration - 1)
        microbatches = math.ceil(len(rows) / config.batch_size)
        total_microbatches = max(1, math.ceil(microbatches * epochs))
        if config.max_steps is not None:
            total_microbatches = (
                config.max_steps + config.add_steps * (iteration - 1)
            ) * config.gradient_accumulation_steps
        total_steps = math.ceil(total_microbatches / config.gradient_accumulation_steps)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            foreach=False,
        )
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.model.train()
        rng = random.Random(config.seed + iteration)

        def batches():
            while True:
                order = list(rows)
                rng.shuffle(order)
                for start in range(0, len(order), config.batch_size):
                    yield order[start : start + config.batch_size]

        stream = batches()
        losses, seen_tokens, seen_examples = [], 0, 0
        started = time.monotonic()
        for step in range(total_steps):
            remaining = total_microbatches - step * config.gradient_accumulation_steps
            group = [
                next(stream) for _ in range(min(config.gradient_accumulation_steps, remaining))
            ]
            # Weight by actual target tokens, including the last short accumulation group.
            target_count = sum(
                sum(t != -100 for t in row["labels"][1:]) for batch in group for row in batch
            )
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            for batch_rows in group:
                batch = collate(batch_rows, self.tokenizer.pad_token_id, self.device)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                )
                loss = loss_sum(outputs.logits, batch["labels"], config.z_loss) / target_count
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite training loss; checkpoint has not been committed"
                    )
                loss.backward()
                step_loss += loss.detach().item()
                seen_examples += len(batch_rows)
                del outputs, loss, batch
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                config.max_grad_norm,
                error_if_nonfinite=True,
                foreach=False,
            )
            optimizer.step()
            losses.append(step_loss)
            seen_tokens += target_count
            if step == 0 or (step + 1) % 10 == 0 or step + 1 == total_steps:
                print(f"  train {step + 1}/{total_steps}: loss={step_loss:.5f}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer
        if config.gradient_checkpointing:
            self.model.gradient_checkpointing_disable()
        self.model.eval()
        return {
            "optimizer_steps": total_steps,
            "examples": len(rows),
            "dropped_overlength": dropped,
            "examples_seen": seen_examples,
            "target_tokens": seen_tokens,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "mean_loss": sum(losses) / len(losses),
            "seconds": time.monotonic() - started,
            "loss_mask": config.loss_mask,
        }

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory, safe_serialization=True)
        self.tokenizer.save_pretrained(directory)

    def close(self):
        del self.model
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()


def doctor(requested="auto"):
    device = select_device(requested)
    # Exercise actual allocation, forward and backward; availability alone is insufficient.
    value = torch.tensor([1.0, 2.0], device=device, requires_grad=True)
    value.square().sum().backward()
    if value.grad.cpu().tolist() != [2.0, 4.0]:
        raise RuntimeError("Device gradient check failed")
    import transformers

    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "macos": platform.mac_ver()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "device": str(device),
        "memory_gib": physical_memory() / 2**30 if physical_memory() else None,
        "forward_backward": "passed",
    }
