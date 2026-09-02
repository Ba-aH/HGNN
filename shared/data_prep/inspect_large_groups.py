import json
import pandas as pd

with open("all_contexts.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

group_sizes = df.groupby("context_group_id").size()

# Look at the largest groups
big_groups = group_sizes[group_sizes >= 9].sort_values(ascending=False)
print("Largest groups (context_group_id -> size):")
print(big_groups)

print("\n" + "=" * 80)
for gid in big_groups.index[:5]:  # inspect top 5 biggest
    sub = df[df["context_group_id"] == gid]
    print(f"\n--- context_group_id = {gid} (size={len(sub)}) ---")
    print("Distinct citing_uri:", sub["citing_uri"].nunique())
    print("Distinct cited_uri:", sub["cited_uri"].nunique())
    print("Distinct context strings:", sub["context"].nunique())
    print("citing_idx values:", sorted(sub["citing_idx"].unique().tolist()))
    print("Sample context (first 150 chars):", sub["context"].iloc[0][:150])
    print("citing_uri values:", sub["citing_uri"].unique()[:5])
