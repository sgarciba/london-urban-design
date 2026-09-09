from pathlib import Path

with open(Path(__file__).resolve().parent / "VERSION") as version_file:
    __version__ = version_file.read().strip()
