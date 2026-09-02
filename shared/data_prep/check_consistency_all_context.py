import json
import pandas as pd

with open("all_contexts_fixed.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)

print("Number of rows:", len(df))

# 1. citation_type breakdown (as labeled in the file)
print("\ncitation_type counts:\n", df["citation_type"].value_counts())
print("\ncitation_type percentages:\n", df["citation_type"].value_counts(normalize=True) * 100)

# 2. cross-check: does context_group_id agree with citation_type?
group_sizes = df.groupby("context_group_id").size()
computed_type = group_sizes.apply(lambda n: "single" if n == 1 else "multiple").sort_index()
label_from_data = df.drop_duplicates("context_group_id").set_index("context_group_id")["citation_type"].sort_index()
mismatch = (computed_type.values != label_from_data.values).sum()
print("\nGroups where citation_type disagrees with actual group size:", mismatch)

# 3. group size distribution
print("\nGroup size distribution:\n", group_sizes.value_counts().sort_index())

# 4. exact duplicate rows (same cited_uri, citing_uri, context)
dup_count = df.duplicated(subset=["citing_uri", "cited_uri", "context"]).sum()
print("\nExact duplicate rows (citing_uri, cited_uri, context):", dup_count)

# 5. duplicated context text reused across different citing_uri (possible boilerplate)
context_counts = df["context"].value_counts()
reused_context = context_counts[context_counts > 1]
print("\nNumber of distinct context strings reused more than once:", len(reused_context))
if len(reused_context) > 0:
    sample_ctx = reused_context.index[0]
    citing_variety = df[df["context"] == sample_ctx]["citing_uri"].nunique()
    print(f"Most duplicated context appears {reused_context.iloc[0]} times, across {citing_variety} distinct citing_uri(s)")