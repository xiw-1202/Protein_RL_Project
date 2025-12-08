"""
Download ProteinGym metadata
This lists all available datasets
"""

import requests
from pathlib import Path


def download_metadata():
    """Download the ProteinGym metadata file"""

    print("=" * 70)
    print("DOWNLOADING PROTEINGYM METADATA")
    print("=" * 70)

    url = "https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv"

    print(f"\nDownloading from: {url}\n")

    # Create directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    output_path = data_dir / "DMS_substitutions_metadata.csv"

    try:
        response = requests.get(url, timeout=30)

        if response.status_code == 200:
            # Save file
            with open(output_path, "wb") as f:
                f.write(response.content)

            # Verify it's valid
            import pandas as pd

            df = pd.read_csv(output_path)

            print(f"✓ Download successful!")
            print(f"  File: {output_path}")
            print(f"  Datasets available: {len(df)}")
            print(f"  Columns: {len(df.columns)}")

            # Show sample
            print(f"\nSample datasets:")
            for i in range(min(5, len(df))):
                print(f"  • {df.iloc[i]['DMS_id']}")

            print("\n" + "=" * 70)
            print("✓ METADATA DOWNLOADED!")
            print("=" * 70)
            print("\nNext: python 1_select_datasets.py")

            return True
        else:
            print(f"✗ Download failed: HTTP {response.status_code}")
            return False

    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    download_metadata()
