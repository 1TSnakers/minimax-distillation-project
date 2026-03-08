import os
import json
import time
import gzip
from datetime import timedelta

import humanize
from datasets import load_dataset
from ollama import Client
from ollama._types import ResponseError


# =========================
# CONFIG
# =========================
MODEL_NAME = "minimax-m2.5:cloud"
DATASET_NAME = "databricks/databricks-dolly-15k"
CACHE_FILE = "datasets/dolly_prompts.jsonl.gz"
OUTPUT_FILE = "teacher_dataset.jsonl"
BATCH_SIZE = 10
WAIT_ON_429 = 600  # 10 minutes


# =========================
# LOAD DATASET (JSONL.GZ)
# =========================
def load_or_cache_dataset():
    os.makedirs("datasets", exist_ok=True)

    if os.path.exists(CACHE_FILE):
        print("Loading cached dataset...")
        prompts = []
        with gzip.open(CACHE_FILE, "rt", encoding="utf-8") as f:
            for line in f:
                prompts.append(json.loads(line))
        return prompts

    print("Downloading dataset from HuggingFace...")
    dataset = load_dataset(DATASET_NAME, split="train")

    print("Caching compressed JSONL dataset...")
    with gzip.open(CACHE_FILE, "wt", encoding="utf-8") as f:
        for row in dataset:
            entry = {
                "instruction": row["instruction"],
                "context": row["context"],
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # reload cleanly
    prompts = []
    with gzip.open(CACHE_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))

    return prompts


# =========================
# LOAD PROCESSED PROMPTS
# =========================
def load_processed_from_output():
    if not os.path.exists(OUTPUT_FILE):
        return set()

    processed = set()

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)

                # Support BOTH legacy and new schema
                if "prompt" in data:
                    processed.add(data["prompt"])
                elif "instruction" in data:
                    processed.add(data["instruction"])

            except:
                continue

    return processed


# =========================
# MAIN
# =========================
def main():
    client = Client()

    prompts = load_or_cache_dataset()
    processed_prompts = load_processed_from_output()

    # Filter remaining prompts
    remaining_prompts = [
        p for p in prompts
        if p["instruction"] not in processed_prompts
    ]

    total_remaining = len(remaining_prompts)

    if total_remaining == 0:
        print("All prompts already processed.")
        return

    print(f"{total_remaining} prompts remaining.")

    start_time = time.time()

    with open(OUTPUT_FILE, "a", encoding="utf-8") as outfile:

        for idx, example in enumerate(remaining_prompts, 1):

            instruction = example["instruction"]
            context = example.get("context", "")

            if context:
                full_prompt = f"{instruction}\n\nContext:\n{context}"
            else:
                full_prompt = instruction

            messages = [{"role": "user", "content": full_prompt}]

            # Retry loop
            while True:
                try:
                    response_text = ""

                    for part in client.chat(
                        MODEL_NAME,
                        messages=messages,
                        stream=True
                    ):
                        response_text += part.message.content

                    break

                except ResponseError as e:
                    if e.status_code == 429:
                        print("Rate limit hit. Waiting 10 minutes...")
                        time.sleep(WAIT_ON_429)
                    else:
                        raise

            # Always write using "prompt" (single canonical format)
            outfile.write(json.dumps({
                "prompt": instruction,
                "response": response_text
            }, ensure_ascii=False) + "\n")

            outfile.flush()

            # Progress report
            if idx % BATCH_SIZE == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / idx
                est_remaining = avg_time * (total_remaining - idx)
                percent_done = (idx / total_remaining) * 100

                print(
                    f"[{idx}/{total_remaining}] "
                    f"{percent_done:.2f}% done | "
                    f"Elapsed: {humanize.naturaldelta(timedelta(seconds=int(elapsed)))} | "
                    f"ETA: {humanize.naturaldelta(timedelta(seconds=int(est_remaining)))}"
                )


if __name__ == "__main__":
    main()