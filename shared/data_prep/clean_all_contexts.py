import json
import pandas as pd

INPUT_FILE = "all_contexts.json"
CLEAN_OUTPUT = "all_contexts_clean.json"
FLAGGED_OUTPUT = "all_contexts_flagged_for_review.json"

MIN_SUSPICIOUS_SIZE = 5  # tune this threshold if needed

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
print("Rows before:", len(df))

# -----------------------------------------------------------------
# Step 1: Flag likely extraction-failure groups
# Pattern: single citing paper, single shared context text, but a
# large number of distinct cited papers -> boilerplate/fallback text
# mistakenly reused as "context" for many references, not a real
# "multiple citations in one sentence" marker.
# -----------------------------------------------------------------
agg = df.groupby("context_group_id").agg(
    group_size=("context_group_id", "size"),
    n_citing_uri=("citing_uri", "nunique"),
    n_cited_uri=("cited_uri", "nunique"),
    n_contexts=("context", "nunique"),
)

suspicious_ids = agg[
    (agg["group_size"] >= MIN_SUSPICIOUS_SIZE)
    & (agg["n_citing_uri"] == 1)
    & (agg["n_contexts"] == 1)
    & (agg["n_cited_uri"] == agg["group_size"])
].index

suspicious_rows = df["context_group_id"].isin(suspicious_ids)
print(f"\nSuspicious groups found: {len(suspicious_ids)}")
print(f"Rows flagged for review: {suspicious_rows.sum()}")

df_flagged = df[suspicious_rows].copy()
df = df[~suspicious_rows].copy()

# -----------------------------------------------------------------
# Step 2: Drop exact duplicate rows (same citing_uri, cited_uri, context)
# -----------------------------------------------------------------
before = len(df)
df = df.drop_duplicates(subset=["citing_uri", "cited_uri", "context"]).reset_index(drop=True)
print(f"\nDropped {before - len(df)} exact duplicate rows")

# -----------------------------------------------------------------
# Step 3: Recompute citation_type from actual context_group_id size
# (ground truth, since it's derived directly from the remaining data)
# -----------------------------------------------------------------
group_sizes = df.groupby("context_group_id")["context_group_id"].transform("size")
correct_type = group_sizes.apply(lambda n: "single" if n == 1 else "multiple")

mismatch_mask = df["citation_type"] != correct_type
print(f"Fixed {mismatch_mask.sum()} rows with wrong citation_type")

df["citation_type"] = correct_type

print("\nFinal citation_type counts:\n", df["citation_type"].value_counts())

# -----------------------------------------------------------------
# Save outputs
# -----------------------------------------------------------------
with open(CLEAN_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

with open(FLAGGED_OUTPUT, "w", encoding="utf-8") as f:
    json.dump(df_flagged.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

print(f"\nSaved {len(df)} clean rows -> {CLEAN_OUTPUT}")
print(f"Saved {len(df_flagged)} flagged rows -> {FLAGGED_OUTPUT}")