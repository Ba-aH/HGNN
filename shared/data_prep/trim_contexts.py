import json
import sys

TARGET_LEN = 408

def trim_center(text, target_len=TARGET_LEN):
    n = len(text)
    if n <= target_len:
        return text
    excess = n - target_len
    left_cut = excess // 2
    right_cut = excess - left_cut  # gives the extra 1 char cut to the right when excess is odd
    return text[left_cut: n - right_cut]

def main(in_path, out_path):
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        records = data.get("records", data)  # fallback, adjust if needed
    else:
        records = data

    for rec in records:
        if "context" in rec and isinstance(rec["context"], str):
            rec["context"] = trim_center(rec["context"], TARGET_LEN)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Processed {len(records)} records. Saved to {out_path}")

if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "all_contexts.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "all_contexts_trimmed.json"
    main(in_path, out_path)