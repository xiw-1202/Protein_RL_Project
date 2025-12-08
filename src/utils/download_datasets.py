"""
Download datasets using the official ProteinGym ZIP file
Then extract only the 6 we selected
"""

import requests
import zipfile
import shutil
from pathlib import Path
import pandas as pd


def download_and_extract_selected():
    """Download full ZIP, extract only selected datasets, clean up"""

    print("=" * 70)
    print("DOWNLOADING DATASETS VIA ZIP FILE")
    print("=" * 70)

    # Check if we have a selection
    selected_path = Path("data/raw/selected_datasets.csv")

    if not selected_path.exists():
        print(f"\n✗ ERROR: {selected_path} not found")
        print("\nFirst run: python 1_select_datasets.py")
        return

    selected = pd.read_csv(selected_path)
    selected_ids = selected["DMS_id"].tolist()

    print(f"\n✓ Found {len(selected_ids)} selected datasets")
    print(
        f"\nWill download full ZIP and extract only these {len(selected_ids)} datasets"
    )
    print("This saves space compared to keeping all 217 datasets!\n")

    # Setup paths
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    VERSION = "v1.3"
    FILENAME = "DMS_ProteinGym_substitutions.zip"
    URL = f"https://marks.hms.harvard.edu/proteingym/ProteinGym_{VERSION}/{FILENAME}"

    zip_path = data_dir / FILENAME
    temp_extract_dir = data_dir / "temp_proteingym"
    final_dir = data_dir / "dms_datasets"

    # Step 1: Download ZIP
    print("Step 1: Downloading ZIP file...")
    print("-" * 70)
    print(f"URL: {URL}")
    print("Size: ~500 MB")
    print("This will take a few minutes...\n")

    try:
        response = requests.get(URL, stream=True, timeout=300)

        if response.status_code == 200:
            total_size = int(response.headers.get("content-length", 0))

            print(f"File size: {total_size / 1024 / 1024:.1f} MB")
            print("Downloading...")

            downloaded = 0
            chunk_size = 8192

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            percent = (downloaded / total_size) * 100
                            mb_downloaded = downloaded / 1024 / 1024
                            mb_total = total_size / 1024 / 1024
                            print(
                                f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)",
                                end="",
                                flush=True,
                            )

            print("\n✓ Download complete!\n")
        else:
            print(f"✗ Download failed: HTTP {response.status_code}")
            return

    except Exception as e:
        print(f"✗ Download error: {e}")
        return

    # Step 2: Extract ZIP to temp directory
    print("Step 2: Extracting ZIP file...")
    print("-" * 70)

    try:
        # Remove temp dir if exists
        if temp_extract_dir.exists():
            shutil.rmtree(temp_extract_dir)

        temp_extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            file_list = zip_ref.namelist()
            total_files = len(file_list)

            print(f"Total files in ZIP: {total_files}")
            print("Extracting to temporary directory...")

            for i, file in enumerate(file_list, 1):
                zip_ref.extract(file, temp_extract_dir)

                if i % 20 == 0 or i == total_files:
                    percent = (i / total_files) * 100
                    print(
                        f"\r  Progress: {percent:.1f}% ({i}/{total_files} files)",
                        end="",
                        flush=True,
                    )

            print("\n✓ Extraction complete!\n")

    except Exception as e:
        print(f"✗ Extraction error: {e}")
        return

    # Step 3: Copy only selected datasets
    print("Step 3: Copying selected datasets...")
    print("-" * 70)

    # Create final directory
    final_dir.mkdir(parents=True, exist_ok=True)

    # Find all CSV files in extracted directory
    extracted_csvs = list(temp_extract_dir.glob("**/*.csv"))

    print(f"Found {len(extracted_csvs)} CSV files in extracted data")
    print(f"Copying {len(selected_ids)} selected datasets...\n")

    copied = []
    not_found = []

    for i, dms_id in enumerate(selected_ids, 1):
        print(f"[{i}/{len(selected_ids)}] {dms_id}")

        # Find this dataset in extracted files
        found = False

        for csv_file in extracted_csvs:
            if csv_file.stem == dms_id:
                # Copy to final directory
                dest = final_dir / f"{dms_id}.csv"
                shutil.copy2(csv_file, dest)

                # Verify
                df = pd.read_csv(dest)
                print(f"  ✓ Copied: {len(df):,} variants")

                copied.append({"dms_id": dms_id, "n_variants": len(df)})

                found = True
                break

        if not found:
            print(f"  ✗ Not found in ZIP")
            not_found.append(dms_id)

    print()

    # Step 4: Clean up
    print("Step 4: Cleaning up...")
    print("-" * 70)

    # Remove ZIP file
    zip_path.unlink()
    print(f"✓ Removed ZIP file: {zip_path}")

    # Remove temp extraction directory
    shutil.rmtree(temp_extract_dir)
    print(f"✓ Removed temp directory: {temp_extract_dir}")

    print()

    # Summary
    print("=" * 70)
    print("DOWNLOAD SUMMARY")
    print("=" * 70)
    print(f"Requested:  {len(selected_ids)}")
    print(f"Copied:     {len(copied)}")
    print(f"Not found:  {len(not_found)}")

    if copied:
        print("\n✓ SUCCESSFULLY COPIED:")
        for item in copied:
            print(f"  • {item['dms_id']}: {item['n_variants']:,} variants")

        # Save summary
        summary_df = pd.DataFrame(copied)
        summary_path = final_dir / "_download_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✓ Summary: {summary_path}")

    if not_found:
        print("\n✗ NOT FOUND:")
        for dms_id in not_found:
            print(f"  • {dms_id}")

    print("\n" + "=" * 70)

    if len(copied) >= 6:
        print("✓✓✓ ALL DATASETS READY! ✓✓✓")
    elif len(copied) >= 3:
        print("✓ ENOUGH TO START!")
    else:
        print("⚠ FEW DATASETS FOUND")

    print("=" * 70)

    if len(copied) >= 3:
        print(f"\nDatasets saved to: {final_dir}")
        print("\nNext: python 3_extract_wild_types.py")


if __name__ == "__main__":
    download_and_extract_selected()
