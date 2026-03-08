# MiniMax Distillation Project

Collects teacher model responses from MiniMax M2.5 on the Databricks Dolly-15k dataset for distillation purposes.

## Goal

The collected responses will be used to fine-tune smaller models like Qwen3-8B (and other models in the future) to replicate MiniMax M2.5's reasoning and outputs.

## Setup

```bash
pip install datasets ollama humanize
```

## Usage

1. Make sure Ollama is running with `minimax-m2.5:cloud` available
2. Run the collector:

```bash
python collect_teacher_hf.py
```

## Output

- `datasets/dolly_prompts.jsonl.gz` — Cached input prompts
- `teacher_dataset.jsonl` — Collected teacher responses (appended incrementally)

The script resumes from where it left off if interrupted.

## Contributing

Contributions are welcome! Feel free to open issues or submit PRs to improve the collection pipeline or add support for additional datasets and target models.

And if you end up finishing the half done teacher dataset, PLEASE OPEN A PR. Currently, the teacher dataset is unfinished, don't use it in production, or in anything right now for that matter.