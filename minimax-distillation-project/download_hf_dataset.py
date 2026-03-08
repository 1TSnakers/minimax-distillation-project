import json
import gzip
import os
from datasets import load_dataset

# --- Config ---
DATASET_NAME = "databricks/databricks-dolly-15k"
PROMPT_COLUMN = "instruction"
OUTPUT_DIR = "datasets"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dolly_prompts.jsonl.gz")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Skip download if already exists
if os.path.exists(OUTPUT_FILE):
    print(f"Using cached dataset at {OUTPUT_FILE}")
else:
    print(f"Downloading {DATASET_NAME} and compressing prompts...")
    dataset = load_dataset(DATASET_NAME, split="train")
    with gzip.open(OUTPUT_FILE, "wt", encoding="utf-8") as f:
        for example in dataset:
            prompt = example[PROMPT_COLUMN].strip()
            if prompt:
                json.dump({"prompt": prompt}, f, ensure_ascii=False)
                f.write("\n")
    print(f"Saved compressed dataset to {OUTPUT_FILE}")