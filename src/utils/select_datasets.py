"""
Select 6 diverse datasets for the project
Picks from different size ranges for variety

Update:
- Add --seed for reproducible selection
- Use pandas.sample(random_state=...) consistently
- Optional --exclude_ids (txt file, one DMS_id per line) to avoid duplicates
"""

import argparse
from pathlib import Path
import pandas as pd


def load_exclude_ids(path_str: str):
    if not path_str:
        return set()
    p = Path(path_str)
    if not p.exists():
        print(f"⚠ Exclude file not found: {p} (ignoring)")
        return set()
    ids = {line.strip() for line in p.read_text().splitlines() if line.strip()}
    return ids


def select_6_datasets(seed: int = 42, exclude_ids=None):
    """Select 6 diverse datasets from metadata"""
    exclude_ids = set(exclude_ids or [])

    print("=" * 70)
    print("SELECTING 6 DIVERSE DATASETS")
    print("=" * 70)
    print(f"Seed: {seed}")
    if exclude_ids:
        print(f"Excluding {len(exclude_ids)} dataset(s)")

    # Load metadata
    metadata_path = Path("data/raw/DMS_substitutions_metadata.csv")
    if not metadata_path.exists():
        print(f"\n✗ ERROR: Metadata not found at {metadata_path}")
        print("\nFirst download metadata:")
        print("  python src/utils/download_metadata.py")
        return 1

    metadata = pd.read_csv(metadata_path)
    print(f"\n✓ Loaded metadata: {len(metadata)} datasets available")

    # Filter: manageable size (500-8000 variants)
    df = metadata[
        (metadata["DMS_total_number_mutants"] >= 500)
        & (metadata["DMS_total_number_mutants"] <= 8000)
    ].copy()

    # Exclude specific IDs if requested
    if exclude_ids:
        df = df[~df["DMS_id"].isin(exclude_ids)].copy()

    print(f"✓ Filtered to 500-8000 variants (after exclude): {len(df)} datasets")

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

    selected = []

    # Use deterministic random_state for reproducibility.
    # Use different offsets so each bin isn't correlated by identical random_state.
    if len(small) >= 2:
        selected.extend(small.sample(n=2, random_state=seed).to_dict("records"))
    elif len(small) > 0:
        selected.extend(small.to_dict("records"))

    if len(medium) >= 2:
        selected.extend(medium.sample(n=2, random_state=seed + 1).to_dict("records"))
    elif len(medium) > 0:
        selected.extend(medium.to_dict("records"))

    if len(large) >= 2:
        selected.extend(large.sample(n=2, random_state=seed + 2).to_dict("records"))
    elif len(large) > 0:
        selected.extend(large.to_dict("records"))

    # If we don't have 6 yet, fill from remaining
    if len(selected) < 6:
        remaining = df[~df["DMS_id"].isin([s["DMS_id"] for s in selected])]
        need = 6 - len(selected)
        if len(remaining) >= need:
            selected.extend(
                remaining.sample(n=need, random_state=seed + 3).to_dict("records")
            )

    # Take first 6
    selected = selected[:6]

    print(f"\n✓ Selected {len(selected)} diverse datasets:\n")

    for i, dataset in enumerate(selected, 1):
        n_vars = dataset["DMS_total_number_mutants"]
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

    return 0


def main():
    parser = argparse.ArgumentParser(description="Select 6 ProteinGym datasets")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--exclude_ids",
        type=str,
        default="",
        help="Path to txt file with DMS_id to exclude (one per line)",
    )
    args = parser.parse_args()

    exclude_ids = load_exclude_ids(args.exclude_ids)
    raise SystemExit(select_6_datasets(seed=args.seed, exclude_ids=exclude_ids))


if __name__ == "__main__":
    main()
