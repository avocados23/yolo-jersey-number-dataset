import json
import os
import requests
from tqdm import tqdm
import time

BASE_DIR = "basketball-jersey-numbers-ocr.v1i.openai"
OUTPUT_DIR = "dataset"

SPLITS = {
    "train": "_annotations.train.jsonl",
    "valid": "_annotations.valid.jsonl",
    "test":  "_annotations.test.jsonl"
}

MAX_RETRIES = 5


def safe_download(url, output_path):
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(output_path, "wb") as f:
                    f.write(r.content)
                return True
        except:
            pass

        time.sleep(1 + attempt)
    return False


def extract_entry(line):
    data = json.loads(line)

    msgs = data["messages"]

    # get user image
    user_msg = next(
        m for m in msgs
        if m["role"] == "user" and isinstance(m["content"], list)
    )
    img_url = user_msg["content"][0]["image_url"]["url"]

    # get assistant label
    label = next(m["content"] for m in msgs if m["role"] == "assistant")
    label = label.strip()

    return img_url, label


def process_jsonl(split_name, filename):
    input_path = os.path.join(BASE_DIR, filename)
    print(f"\n=== Processing {split_name}: {input_path} ===")

    # read lines
    with open(input_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    # prepare output dirs
    split_output_dir = os.path.join(OUTPUT_DIR, split_name)
    images_root = split_output_dir  # classification uses folder-per-class

    for idx, line in enumerate(tqdm(lines, desc=split_name), start=1):
        try:
            url, label = extract_entry(line)
        except Exception as e:
            print(f"Bad JSONL line {idx}: {e}")
            continue

        # class folder
        class_dir = os.path.join(images_root, label)
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, f"{idx:06d}.jpg")

        if not safe_download(url, save_path):
            print(f"Failed download: {url}")


def main():
    for split, file in SPLITS.items():
        process_jsonl(split, file)


if __name__ == "__main__":
    main()
