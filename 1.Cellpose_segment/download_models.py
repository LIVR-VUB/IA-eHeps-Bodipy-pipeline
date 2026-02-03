import os
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Error: 'huggingface_hub' is not installed.")
    print("Please install it running: pip install huggingface_hub")
    sys.exit(1)

# Configuration
REPO_ID = "LIVR-VUB/eHeps-Bodipy-models"
MODELS = [
    "cpsam_eHeps_nuclei",
    "cpsam_eHeps_v3"
]
OUTPUT_DIR = "models"

def download_models():
    print("==========================================")
    print(f" Downloading models from: {REPO_ID}")
    print("==========================================\n")

    # Create output directory
    if not os.path.exists(OUTPUT_DIR):
        print(f" Creating directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)
    
    for filename in MODELS:
        print(f" Downloading: {filename}...")
        try:
            # Download file
            file_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=filename,
                local_dir=OUTPUT_DIR,
                local_dir_use_symlinks=False  # Download actual file, not symlink
            )
            print(f"  ✓ Success! Saved to: {file_path}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            sys.exit(1)

    print("\n==========================================")
    print(" All downloads complete!")
    print(f" Models are in: {os.path.abspath(OUTPUT_DIR)}")
    print("==========================================")

if __name__ == "__main__":
    download_models()
