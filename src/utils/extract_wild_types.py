"""
Extract wild-type sequences from downloaded datasets
"""

import pandas as pd
from pathlib import Path
from collections import Counter


def extract_wild_type(csv_path):
    """Extract wild-type sequence from a dataset"""

    df = pd.read_csv(csv_path)

    # Method 1: Look for explicit WT row
    if "mutant" in df.columns:
        wt_rows = df[df["mutant"].isin(["WT", "", "wt"])]

        if len(wt_rows) > 0 and "mutated_sequence" in df.columns:
            wt_seq = wt_rows.iloc[0]["mutated_sequence"]
            if isinstance(wt_seq, str) and len(wt_seq) > 0:
                return wt_seq, "explicit_WT"

    # Method 2: Most common sequence
    if "mutated_sequence" in df.columns:
        sequences = df["mutated_sequence"].dropna()

        if len(sequences) > 0:
            seq_counts = Counter(sequences)
            most_common = seq_counts.most_common(1)[0][0]

            if isinstance(most_common, str) and len(most_common) > 0:
                return most_common, "most_common"

    # Method 3: Reconstruct from single mutants
    if "mutant" in df.columns:
        import re

        single_mutants = df[df["mutant"].str.match(r"^[A-Z]\d+[A-Z]$", na=False)]

        if len(single_mutants) > 10:
            positions = {}

            for mut in single_mutants["mutant"]:
                try:
                    wt_aa = mut[0]
                    pos = int(mut[1:-1]) - 1
                    positions[pos] = wt_aa
                except:
                    continue

            if len(positions) > 20:
                max_pos = max(positions.keys())
                wt_seq = ["X"] * (max_pos + 1)

                for pos, aa in positions.items():
                    wt_seq[pos] = aa

                wt_seq = "".join(wt_seq)

                # Accept if less than 10% unknown positions
                if wt_seq.count("X") < len(wt_seq) * 0.1:
                    return wt_seq, "reconstructed"

    return None, "failed"


def extract_all():
    """Extract wild-types for all downloaded datasets"""

    print("=" * 70)
    print("EXTRACTING WILD-TYPE SEQUENCES")
    print("=" * 70)

    # Find downloaded datasets
    data_dir = Path("data/raw/dms_datasets")

    if not data_dir.exists():
        print(f"\n✗ ERROR: {data_dir} not found")
        print("\nFirst run: python 2_download_datasets.py")
        return

    csv_files = [f for f in data_dir.glob("*.csv") if not f.name.startswith("_")]

    if len(csv_files) == 0:
        print(f"\n✗ ERROR: No datasets found")
        print("\nFirst run: python 2_download_datasets.py")
        return

    print(f"\n✓ Found {len(csv_files)} dataset(s)\n")

    # Create output directory
    wt_dir = Path("data/raw/wild_types")
    wt_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for i, csv_file in enumerate(sorted(csv_files), 1):
        dms_id = csv_file.stem

        print(f"[{i}/{len(csv_files)}] {dms_id}")

        wt_seq, method = extract_wild_type(csv_file)

        if wt_seq:
            print(f"  ✓ Extracted ({method})")
            print(f"    Length: {len(wt_seq)} amino acids")
            print(f"    First 50: {wt_seq[:50]}...")

            # Save as FASTA
            fasta_path = wt_dir / f"{dms_id}.fasta"

            with open(fasta_path, "w") as f:
                f.write(f">{dms_id}\n")
                # Write in lines of 60
                for j in range(0, len(wt_seq), 60):
                    f.write(wt_seq[j : j + 60] + "\n")

            results.append(
                {
                    "dms_id": dms_id,
                    "length": len(wt_seq),
                    "method": method,
                    "fasta_path": str(fasta_path),
                    "sequence": wt_seq,
                }
            )
        else:
            print(f"  ✗ Failed to extract")

        print()

    # Summary
    print("=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total datasets:  {len(csv_files)}")
    print(f"Extracted:       {len(results)}")
    print(f"Failed:          {len(csv_files) - len(results)}")

    if results:
        lengths = [r["length"] for r in results]
        print(f"\nSequence lengths:")
        print(f"  Shortest: {min(lengths)} AA")
        print(f"  Longest:  {max(lengths)} AA")
        print(f"  Average:  {sum(lengths) // len(lengths)} AA")

        # Save summary
        results_df = pd.DataFrame(results)
        summary_path = wt_dir / "_wild_type_summary.csv"
        results_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary: {summary_path}")

    print("\n" + "=" * 70)

    if len(results) >= 6:
        print("✓✓✓ ALL WILD-TYPES EXTRACTED! ✓✓✓")
    elif len(results) >= 3:
        print("✓ ENOUGH TO START!")
    else:
        print("⚠ FEW WILD-TYPES EXTRACTED")

    print("=" * 70)

    if len(results) >= 3:
        print("\n✓✓✓ DATA SETUP COMPLETE! ✓✓✓")
        print("\nYou now have:")
        for r in results:
            print(f"  • {r['dms_id']}: {r['length']} AA")

        print(f"\nFiles saved:")
        print(f"  Datasets:    data/raw/dms_datasets/")
        print(f"  Wild-types:  data/raw/wild_types/")

        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\n1. Validate Oracle Quality")
        print("   Compute ESM-2 correlation with experimental fitness")
        print("   This determines oracle quality tiers (HIGH/MEDIUM/LOW)")
        print("\n2. Implement Baselines")
        print("   - Random sampling")
        print("   - Greedy hill climbing")
        print("   - Simulated annealing")
        print("\n3. Implement RL Methods")
        print("   - Contextual bandits")
        print("   - PPO (optional)")
        print("\n4. Run Experiments")
        print("   Compare all methods across oracle quality tiers")


if __name__ == "__main__":
    extract_all()
