"""
Complete setup for balanced 2-2-2 distribution
Creates: balanced_datasets.csv and balanced_ids.txt
Updates: _wild_type_summary.csv
"""

import requests
import zipfile
import shutil
from pathlib import Path
import pandas as pd
from collections import Counter


def extract_wild_type(csv_path):
    """Extract wild-type from dataset"""
    df = pd.read_csv(csv_path)

    # Method 1: Explicit WT
    if "mutant" in df.columns:
        wt_rows = df[df["mutant"].isin(["WT", "", "wt"])]
        if len(wt_rows) > 0 and "mutated_sequence" in df.columns:
            wt_seq = wt_rows.iloc[0]["mutated_sequence"]
            if isinstance(wt_seq, str) and len(wt_seq) > 0:
                return wt_seq, "explicit"

    # Method 2: Most common sequence
    if "mutated_sequence" in df.columns:
        sequences = df["mutated_sequence"].dropna()
        if len(sequences) > 0:
            most_common = Counter(sequences).most_common(1)[0][0]
            if isinstance(most_common, str) and len(most_common) > 0:
                return most_common, "common"

    return None, "failed"


def setup_balanced_datasets():
    """Complete setup for 2-2-2 distribution"""

    print("=" * 70)
    print("SETUP BALANCED DATASET DISTRIBUTION (2-2-2)")
    print("=" * 70)

    # Define what we're doing
    new_datasets = [
        {"id": "CBPA2_HUMAN_Tsuboyama_2023_1O6X", "tier": "HIGH", "rho": 0.690},
        {"id": "SAV1_MOUSE_Tsuboyama_2023_2YSB", "tier": "MEDIUM", "rho": 0.566},
    ]

    remove_datasets = [
        "DN7A_SACS2_Tsuboyama_2023_1JIC",
        "RFAH_ECOLI_Tsuboyama_2023_2LCL",
    ]

    final_6 = [
        {"id": "PITX2_HUMAN_Tsuboyama_2023_2L7M", "tier": "HIGH", "rho": 0.689},
        {"id": "CBPA2_HUMAN_Tsuboyama_2023_1O6X", "tier": "HIGH", "rho": 0.690},
        {"id": "SRC_HUMAN_Ahler_2019", "tier": "MEDIUM", "rho": 0.474},
        {"id": "SAV1_MOUSE_Tsuboyama_2023_2YSB", "tier": "MEDIUM", "rho": 0.566},
        {"id": "PAI1_HUMAN_Huttinger_2021", "tier": "LOW", "rho": 0.393},
        {"id": "CCR5_HUMAN_Gill_2023", "tier": "LOW", "rho": 0.349},
    ]

    print("\nPlan:")
    print("  + Add: CBPA2_HUMAN (HIGH), SAV1_MOUSE (MEDIUM)")
    print("  - Remove: DN7A_SACS2 (LOW), RFAH_ECOLI (LOW)")
    print("  = Result: 2 HIGH, 2 MEDIUM, 2 LOW")
    print("  → Creates: balanced_datasets.csv, balanced_ids.txt")
    print("  → Updates: _wild_type_summary.csv\n")

    data_dir = Path("data/raw")
    dms_dir = data_dir / "dms_datasets"
    wt_dir = data_dir / "wild_types"

    # STEP 1: Download new datasets
    print("=" * 70)
    print("STEP 1: Download 2 new datasets")
    print("=" * 70)

    URL = "https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip"
    zip_path = data_dir / "temp.zip"
    temp_dir = data_dir / "temp"

    print("\nDownloading ProteinGym ZIP (~500 MB)...")
    response = requests.get(URL, stream=True, timeout=300)
    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                print(
                    f"\r  {downloaded/total*100:.1f}% ({downloaded/1024/1024:.1f}/{total/1024/1024:.1f} MB)",
                    end="",
                )

    print("\n✓ Downloaded")

    # Extract
    print("\nExtracting...")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    with zipfile.ZipFile(zip_path, "r") as z:
        for f in z.namelist():
            if any(d["id"] in f for d in new_datasets):
                z.extract(f, temp_dir)

    # Copy
    for ds in new_datasets:
        for csv in temp_dir.glob("**/*.csv"):
            if ds["id"] in csv.name:
                dest = dms_dir / csv.name
                shutil.copy2(csv, dest)
                df = pd.read_csv(dest)
                print(f"  ✓ {csv.name}: {len(df):,} variants")

    zip_path.unlink()
    shutil.rmtree(temp_dir)

    # STEP 2: Extract wild-types for new datasets
    print("\n" + "=" * 70)
    print("STEP 2: Extract wild-types for new datasets")
    print("=" * 70 + "\n")

    new_wt_records = []

    for ds in new_datasets:
        csv_path = dms_dir / f"{ds['id']}.csv"

        if csv_path.exists():
            wt_seq, method = extract_wild_type(csv_path)

            if wt_seq:
                fasta_path = wt_dir / f"{ds['id']}.fasta"
                with open(fasta_path, "w") as f:
                    f.write(f">{ds['id']}\n")
                    for i in range(0, len(wt_seq), 60):
                        f.write(wt_seq[i : i + 60] + "\n")

                print(f"✓ {ds['id']}: {len(wt_seq)} AA ({method})")

                new_wt_records.append(
                    {
                        "dms_id": ds["id"],
                        "length": len(wt_seq),
                        "method": method,
                        "sequence": wt_seq,
                    }
                )

    # STEP 3: Remove old datasets
    print("\n" + "=" * 70)
    print("STEP 3: Remove 2 old LOW datasets")
    print("=" * 70 + "\n")

    for ds_id in remove_datasets:
        # Remove dataset
        csv_path = dms_dir / f"{ds_id}.csv"
        if csv_path.exists():
            csv_path.unlink()
            print(f"✓ Removed: {ds_id}.csv")

        # Remove wild-type
        fasta_path = wt_dir / f"{ds_id}.fasta"
        if fasta_path.exists():
            fasta_path.unlink()
            print(f"✓ Removed: {ds_id}.fasta")

    # STEP 4: Update wild-type summary
    print("\n" + "=" * 70)
    print("STEP 4: Update wild-type summary")
    print("=" * 70 + "\n")

    summary_path = wt_dir / "_wild_type_summary.csv"

    if summary_path.exists():
        # Load existing
        summary_df = pd.read_csv(summary_path)

        # Remove old datasets
        summary_df = summary_df[~summary_df["dms_id"].isin(remove_datasets)]
        print(f"✓ Removed {len(remove_datasets)} old entries")

        # Add new datasets
        new_df = pd.DataFrame(new_wt_records)
        summary_df = pd.concat([summary_df, new_df], ignore_index=True)
        print(f"✓ Added {len(new_wt_records)} new entries")

        # Save
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Updated: {summary_path.name}")
    else:
        # Create new
        summary_df = pd.DataFrame(new_wt_records)
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Created: {summary_path.name}")

    # STEP 5: Create NEW balanced selection files
    print("\n" + "=" * 70)
    print("STEP 5: Create balanced selection files")
    print("=" * 70 + "\n")

    # Get variant counts
    final_with_counts = []
    for ds in final_6:
        csv_path = dms_dir / f"{ds['id']}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            final_with_counts.append(
                {
                    "DMS_id": ds["id"],
                    "tier": ds["tier"],
                    "spearman_rho": ds["rho"],
                    "DMS_total_number_mutants": len(df),
                }
            )

    # Save balanced_datasets.csv
    df_selection = pd.DataFrame(final_with_counts)
    balanced_csv = data_dir / "balanced_datasets.csv"
    df_selection.to_csv(balanced_csv, index=False)
    print(f"✓ Created: {balanced_csv.name}")

    # Save balanced_ids.txt
    balanced_txt = data_dir / "balanced_ids.txt"
    with open(balanced_txt, "w") as f:
        for ds in final_6:
            f.write(ds["id"] + "\n")
    print(f"✓ Created: {balanced_txt.name}")

    # SUMMARY
    print("\n" + "=" * 70)
    print("✓✓✓ SETUP COMPLETE!")
    print("=" * 70)

    print("\nFiles created/updated:")
    print(f"  • {balanced_csv.name} (NEW)")
    print(f"  • {balanced_txt.name} (NEW)")
    print(f"  • _wild_type_summary.csv (UPDATED)")

    print("\nOld files preserved:")
    print("  • selected_datasets.csv (original selection)")
    print("  • selected_ids.txt (original selection)")

    print("\nFinal balanced distribution:")
    for tier in ["HIGH", "MEDIUM", "LOW"]:
        tier_data = [d for d in final_6 if d["tier"] == tier]
        print(f"\n{tier} Quality ({len(tier_data)} datasets):")
        for d in tier_data:
            counts = next(
                (
                    x["DMS_total_number_mutants"]
                    for x in final_with_counts
                    if x["DMS_id"] == d["id"]
                ),
                0,
            )
            print(f"  • {d['id'].split('_')[0]}_{d['id'].split('_')[1]}")
            print(f"    ρ = {d['rho']:.3f}, {counts:,} variants")

    print("\n" + "=" * 70)
    print("Ready for experiments!")
    print("  5 methods × 5 seeds × 4 k × 6 datasets = 600 runs")
    print("=" * 70)


if __name__ == "__main__":
    setup_balanced_datasets()
