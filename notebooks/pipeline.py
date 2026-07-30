import argparse
import sys
sys.path.append("../")

from load_dataset import download_landsat, download_buildings_osmnx
from data_processing import run as run_data_processing
from config import GOOGLE_DRIVE_CLIENT_SECRETS_JSON, FOLDER_NAME, FILE_NAME, ADMINISTRATIVE_AREA, KEEP_COLS

# Run from inside notebooks/ — every step here (like the scripts it wraps)
# resolves data paths relative to that working directory.


def download_landsat_step():
    download_landsat(GOOGLE_DRIVE_CLIENT_SECRETS_JSON, FOLDER_NAME, FILE_NAME)


def download_buildings_step():
    download_buildings_osmnx(ADMINISTRATIVE_AREA, KEEP_COLS)


def process():
    run_data_processing()


# "train" and "predict" steps will be added here once train.py/predict.py
# exist. train_model.py stays out of the pipeline — it's the exploratory
# model-selection tool used to decide what config train.py should use, not
# a reproducible step with a persisted output.
STAGES = {
    "download_landsat": download_landsat_step,
    "download_buildings": download_buildings_step,
    "process": process,
}


def main():
    parser = argparse.ArgumentParser(description="Run the London urban design data pipeline.")
    parser.add_argument(
        "--steps",
        default="process",
        help=(
            f"Comma-separated steps to run, in order. Available: {', '.join(STAGES)}. "
            "Default: process. The download_* steps require an interactive Google OAuth "
            "login and re-fetch large files, so they're opt-in rather than run by default."
        ),
    )
    args = parser.parse_args()

    steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in steps if s not in STAGES]
    if unknown:
        parser.error(f"Unknown step(s): {', '.join(unknown)}. Available: {', '.join(STAGES)}")

    for step in steps:
        print(f"\n=== Running step: {step} ===")
        STAGES[step]()


if __name__ == "__main__":
    main()
