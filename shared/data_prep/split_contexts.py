import json

with open("all_contexts_clean.json", "r", encoding="utf-8") as f:
    data = json.load(f)

single = [r for r in data if r.get("citation_type") == "single"]
multiple = [r for r in data if r.get("citation_type") == "multiple"]

with open("single_contexts.json", "w", encoding="utf-8") as f:
    json.dump(single, f, ensure_ascii=False, indent=2)

with open("multiple_contexts.json", "w", encoding="utf-8") as f:
    json.dump(multiple, f, ensure_ascii=False, indent=2)

print(f"single: {len(single)} records -> single_contexts.json")
print(f"multiple: {len(multiple)} records -> multiple_contexts.json")
