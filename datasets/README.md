# 数据

`commonsenseqa/` 和 `gsm/` 保留原仓库自带的 JSONL 及 few-shot 提示，内容未改动。训练集会按配置的 seed 打乱后选取 `train_samples` 条；验证集按原顺序选取 `eval_samples` 条。

- `commonsenseqa/train_rand_split.jsonl` / `dev_rand_split.jsonl`：使用 `answerKey` 验证最终选项。
- `gsm/train_rand_split.jsonl` / `dev_rand_split.jsonl`：只从 `answer` 的最后一个 `####` 提取正确答案；人工参考推理不进入自生成训练集。
- `prompts*.txt`：原作者提供的少量推理示例、答案提示示例及 direct 基线示例。
- `test_rand_split_no_answers.jsonl`：保留原始无标签测试集，不用于当前有监督答案验证流程。

生成 arithmetic 数据：

```bash
star prepare-arithmetic --data-root datasets --train-count 200 --eval-count 50 --max-digits 3
```

会写入 `arithmetic/` 的训练集、验证集、3 个独立的 few-shot 示例以及带提示的示例。三部分题目不重叠，数字位数为 1 到 `max_digits`。数据中的 `rationale` 仅供检查生成器；加载训练问题时不使用它。生成目录已被 Git 忽略，重复生成需要选择新目录，防止覆盖既有实验数据。

也可在配置中使用 `train_file` / `eval_file` 指向相同格式的自定义 JSONL。配置内所有数据路径相对于该配置文件解析。
