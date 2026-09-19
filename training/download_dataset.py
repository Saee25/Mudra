import os
import sys
import argparse
import zipfile
import shutil
from pathlib import Path
from PIL import Image

# Setup paths relative to script
SCRIPT_DIR = Path(__file__).parent.resolve()
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
TEMP_DIR = RAW_DIR / "temp_extract"
FINAL_DIR = DATA_DIR / "isl_alphabet"
CORRUPT_DIR = RAW_DIR / "corrupt"

def find_class_level(temp_path):
    """
    Finds the directory inside temp_path that contains the class folders (1-9, A-Z).
    Returns the path to that directory, or None if not found.
    """
    valid_classes = set([str(i) for i in range(1, 10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)])
    
    for root, dirs, files in os.walk(temp_path):
        # Check if the directories here look like class names
        upper_dirs = [d.upper() for d in dirs]
        # If we have a decent overlap with valid classes, this is the level
        if len(set(upper_dirs).intersection(valid_classes)) > 10:
            return Path(root)
    return None

def extract_and_normalize(zip_path, force=False):
    # Check if directory exists and has subdirectories (which would be our class folders)
    if FINAL_DIR.exists() and any(x.is_dir() for x in FINAL_DIR.iterdir()) and not force:
        print(f"Dataset already exists in {FINAL_DIR}. Use --force to re-extract.")
        return

    if not zip_path.exists():
        print(f"ERROR: Zip file not found at {zip_path}")
        return

    print(f"Extracting {zip_path} to temporary directory...")
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
    except zipfile.BadZipFile:
        print(f"ERROR: The file at {zip_path} is corrupted or not a valid zip file.")
        shutil.rmtree(TEMP_DIR)
        return
    except Exception as e:
        print(f"ERROR: Extraction failed. Ensure you have enough disk space. Details: {e}")
        shutil.rmtree(TEMP_DIR)
        return

    print("Finding class folders...")
    class_level = find_class_level(TEMP_DIR)
    if not class_level:
        print("ERROR: Could not find class folders (e.g., A-Z, 1-9) in the zip file.")
        shutil.rmtree(TEMP_DIR)
        return

    print(f"Found class level at: {class_level}")
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    CORRUPT_DIR.mkdir(parents=True, exist_ok=True)

    valid_classes = set([str(i) for i in range(1, 10)] + [chr(i) for i in range(ord('A'), ord('Z')+1)])
    
    class_counts = {}
    corrupted_count = 0

    print("Moving and validating images...")
    for class_folder in class_level.iterdir():
        if not class_folder.is_dir():
            continue
            
        class_name = class_folder.name.upper()
        if class_name not in valid_classes:
            # Skip junk folders like __MACOSX
            continue
            
        dest_class_folder = FINAL_DIR / class_name
        dest_class_folder.mkdir(exist_ok=True)
        
        count = 0
        for file_path in class_folder.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Validate image
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    # Move to final destination
                    dest_file = dest_class_folder / file_path.name
                    shutil.move(str(file_path), str(dest_file))
                    count += 1
                except Exception:
                    # Corrupted
                    dest_corrupt = CORRUPT_DIR / f"{class_name}_{file_path.name}"
                    shutil.move(str(file_path), str(dest_corrupt))
                    corrupted_count += 1

        if count > 0:
            class_counts[class_name] = count
            
    print("\nCleaning up temporary files...")
    shutil.rmtree(TEMP_DIR)
    os.remove(zip_path)
    print(f"Deleted {zip_path} and temporary extraction folder.")

    # Summary
    classes_found = sorted(list(class_counts.keys()))
    total_images = sum(class_counts.values())
    print("\n--- Extraction Summary ---")
    print(f"Total Classes Found: {len(classes_found)}")
    print(f"Classes: {', '.join(classes_found)}")
    print(f"Total Images: {total_images}")
    if corrupted_count > 0:
        print(f"Found and skipped {corrupted_count} corrupted images (saved to {CORRUPT_DIR}).")
    print(f"Dataset successfully prepared in {FINAL_DIR}")

def download_via_api():
    try:
        import kaggle
    except OSError as e:
        print("ERROR: Kaggle credentials not found.")
        print("Please place your kaggle.json file in the correct location or set KAGGLE_USERNAME and KAGGLE_KEY environment variables.")
        return None
    except ImportError:
        print("ERROR: kaggle Python package not installed.")
        return None

    DATASET = "prathumarikeri/indian-sign-language-isl"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset {DATASET} via Kaggle API...")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(DATASET, path=str(RAW_DIR), unzip=False)
    
    # Assume the downloaded zip name
    zip_path = RAW_DIR / "indian-sign-language-isl.zip"
    if not zip_path.exists():
        # Kaggle sometimes downloads with a different name based on the slug
        zips = list(RAW_DIR.glob("*.zip"))
        if zips:
            zip_path = zips[0]
        else:
            print("ERROR: Downloaded zip file not found.")
            return None
            
    print(f"Download complete: {zip_path}")
    return zip_path

def main():
    parser = argparse.ArgumentParser(description="Download and extract the ISL dataset.")
    parser.add_argument("--from-zip", type=str, help="Path to a manually downloaded zip file.")
    parser.add_argument("--force", action="store_true", help="Force re-extraction if dataset already exists.")
    args = parser.parse_args()

    if args.from_zip:
        zip_path = Path(args.from_zip)
        if not zip_path.is_absolute():
            zip_path = Path.cwd() / zip_path
        extract_and_normalize(zip_path, force=args.force)
    else:
        print("No --from-zip provided. Attempting to download via Kaggle API...")
        zip_path = download_via_api()
        if zip_path:
            extract_and_normalize(zip_path, force=args.force)

if __name__ == "__main__":
    main()
