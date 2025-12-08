"""
Select 6 diverse datasets for the project
Picks from different size ranges for variety
"""

import pandas as pd
from pathlib import Path
import random


def select_6_datasets():
    """Select 6 diverse datasets from metadata"""

    print("=" * 70)
    print("SELECTING 6 DIVERSE DATASETS")
    print("=" * 70)

    # Load metadata
    metadata_path = Path("data/raw/DMS_substitutions_metadata.csv")

    if not metadata_path.exists():
        print(f"\n✗ ERROR: Metadata not found at {metadata_path}")
        print("\nFirst download metadata:")
        print("  cd data/raw")
        print(
            "  curl -o DMS_substitutions_metadata.csv https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"
        )
        return

    metadata = pd.read_csv(metadata_path)

    print(f"\n✓ Loaded metadata: {len(metadata)} datasets available")

    # Filter: manageable size (500-8000 variants)
    # Not too small (unreliable), not too large (slow)
    df = metadata[
        (metadata["DMS_total_number_mutants"] >= 500)
        & (metadata["DMS_total_number_mutants"] <= 8000)
    ].copy()

    print(f"✓ Filtered to 500-8000 variants: {len(df)} datasets")

    # Stratify by size to get variety
    # Small: 500-1500, Medium: 1500-4000, Large: 4000-8000
    small = df[df["DMS_total_number_mutants"] < 1500]
    medium = df[
        (df["DMS_total_number_mutants"] >= 1500)
        & (df["DMS_total_number_mutants"] < 4000)
    ]
    large = df[df["DMS_total_number_mutants"] >= 4000]

    print(f"\nSize distribution:")
    print(f"  Small (500-1500):    {len(small)} datasets")
    print(f"  Medium (1500-4000):  {len(medium)} datasets")
    print(f"  Large (4000-8000):   {len(large)} datasets")

    # Select 2 from each category
    random.seed(42)  # For reproducibility

    selected = []

    # From small: pick 2 randomly
    if len(small) >= 2:
        selected.extend(small.sample(n=2).to_dict("records"))
    elif len(small) > 0:
        selected.extend(small.to_dict("records"))

    # From medium: pick 2 randomly
    if len(medium) >= 2:
        selected.extend(medium.sample(n=2).to_dict("records"))
    elif len(medium) > 0:
        selected.extend(medium.to_dict("records"))

    # From large: pick 2 randomly
    if len(large) >= 2:
        selected.extend(large.sample(n=2).to_dict("records"))
    elif len(large) > 0:
        selected.extend(large.to_dict("records"))

    # If we don't have 6 yet, fill from remaining
    if len(selected) < 6:
        remaining = df[~df["DMS_id"].isin([s["DMS_id"] for s in selected])]
        need = 6 - len(selected)
        if len(remaining) >= need:
            selected.extend(remaining.sample(n=need).to_dict("records"))

    # Take first 6
    selected = selected[:6]

    print(f"\n✓ Selected {len(selected)} diverse datasets:")
    print()

    for i, dataset in enumerate(selected, 1):
        n_vars = dataset["DMS_total_number_mutants"]

        # Categorize
        if n_vars < 1500:
            size_cat = "SMALL"
        elif n_vars < 4000:
            size_cat = "MEDIUM"
        else:
            size_cat = "LARGE"

        print(f"  {i}. [{size_cat:6s}] {dataset['DMS_id']}")
        print(f"     Variants: {n_vars:,}")

    # Save selection
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_df = pd.DataFrame(selected)
    output_path = output_dir / "selected_datasets.csv"

    selected_df[["DMS_id", "DMS_total_number_mutants"]].to_csv(output_path, index=False)

    print(f"\n✓ Saved selection to: {output_path}")

    # Also save just the IDs
    ids_path = output_dir / "selected_ids.txt"
    with open(ids_path, "w") as f:
        for dataset in selected:
            f.write(f"{dataset['DMS_id']}\n")

    print(f"✓ Saved IDs to: {ids_path}")

    print("\n" + "=" * 70)
    print("✓ SELECTION COMPLETE")
    print("=" * 70)
    print("\nNext: python 2_download_datasets.py")


if __name__ == "__main__":
    select_6_datasets()
