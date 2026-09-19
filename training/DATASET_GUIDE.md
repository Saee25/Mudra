# Mudra — Dataset Guide

This guide explains how to download the Kaggle "Indian Sign Language (ISL)" dataset and place it correctly for the Mudra training pipeline.

## Dataset Details
- **Dataset Link**: [Indian Sign Language (ISL) by Prathum Arikeri](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl)
- **License**: Unknown/Custom (Check Kaggle dataset page). Please ensure you have rights to use this dataset for your purposes.
- **Attribution**: "Dataset: Indian Sign Language (ISL) by Prathum Arikeri, retrieved from Kaggle" (Add this to the project README).
- **Disk Space Needed**:
  - The downloaded zip is approx ~600-800 MB.
  - The extracted dataset will take up another ~1.5 GB.
  - Generating crops and landmarks will take an additional ~2-3 GB.
  - **Recommendation**: Have at least **5 GB** of free disk space.

---

## Method A: Manual Browser Download (Simplest)
This is the easiest method and requires no API setup.

1. Create a Kaggle account or sign in at [kaggle.com](https://www.kaggle.com/).
2. Navigate to the [dataset page](https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl).
3. Click the **Download** button.
4. Move the downloaded `.zip` file into exactly this folder: `mudra/training/data/raw/` (You may need to create the `raw` folder if it doesn't exist). The file name can be anything.
5. Open your terminal, activate your Python virtual environment inside the `training/` folder, and run:
   ```bash
   python download_dataset.py --from-zip data/raw/<your-downloaded-zip-name>.zip
   ```

---

## Method B: Kaggle API (Automatic)
If you prefer a programmatic approach without using the browser:

1. **Create an API Token**:
   - Go to Kaggle, click on your profile picture in the top right, and select **Settings**.
   - Scroll down to the **API** section and click **Create New Token**.
   - This will download a `kaggle.json` file containing your API credentials.

2. **Place the Token**:
   - **Windows**: Place `kaggle.json` in `C:\Users\<Your-Username>\.kaggle\kaggle.json`
   - **macOS / Linux**: Place `kaggle.json` in `~/.kaggle/kaggle.json`

3. **File Permissions (macOS/Linux only)**:
   Ensure your token is private by running this in your terminal:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. **Alternative: Environment Variables**:
   Instead of using the file, you can set the credentials as environment variables:
   - `KAGGLE_USERNAME` = (your username)
   - `KAGGLE_KEY` = (your token key)

5. **Download via Script**:
   Activate your virtual environment inside the `training/` folder and run:
   ```bash
   python download_dataset.py
   ```

---

## Expected Folder Layout Inside Zip
When the zip is extracted, it may contain an extra wrapper folder or junk files (like `__MACOSX` or `.DS_Store`). The extraction script will automatically find the actual class folders (e.g., `1`, `2`, `A`, `B`) and discard the rest.

## Final Expected Layout (After Extraction)
The extraction script ensures your final layout looks exactly like this:
```
training/
└── data/
    └── isl_alphabet/
        ├── 1/
        │   ├── image1.jpg
        │   └── ...
        ├── 2/
        ├── ...
        ├── A/
        ├── B/
        └── Z/
```
You should expect 35 classes (1-9, A-Z) with roughly 1,200 images each, totaling around ~42,000 images.

---

## How to Verify It Worked
To verify the extraction was successful, run:
```bash
python explore_dataset.py
```
**Expected Output**:
- The script should print a summary of the classes, the total number of images, and run a MediaPipe detection gate.
- It will create several charts (like `class_distribution.png` and `sample_grid.png`) in the `outputs/` folder.
- If everything passes, it will print a "PASS" verdict for the MediaPipe detection gate.

---

## Troubleshooting

- **401/403 Errors (Kaggle API)**: Your token is expired or invalid. Go to Kaggle, revoke the old token, and create a new one. Replace the `kaggle.json` file.
- **`kaggle.json` not found**: Ensure the file is placed in exactly the correct `.kaggle` folder for your OS (see Method B).
- **Zip already exists / partial download**: If the download is stuck or corrupted, delete the zip in `data/raw/` and try again.
- **"Running from the wrong folder"**: Ensure your terminal is `cd`'d into the `training/` folder before running scripts, not the root `mudra/` folder.
- **Windows Long Path Issues**: Windows has a 260-character path limit. The extraction script uses relative paths to mitigate this, but if you still see `PathTooLongException`, try moving the entire `mudra` project closer to the root of your drive (e.g., `C:\Mudra\`).
- **Antivirus Slowing Extraction**: Extracting ~40k files can take time. If it's unusually slow on Windows, temporarily pause Windows Defender real-time protection, but remember to turn it back on.
- **Not Enough Disk Space**: If the extraction crashes midway due to lack of space, free up at least 5GB, delete the `data/isl_alphabet/` folder, and re-run with `--force`.

---

## Version Control
**Do not commit the `data/` folder.** Make sure `training/data/` is excluded in your `.gitignore`.
