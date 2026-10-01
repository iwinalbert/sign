from __future__ import annotations

import csv
from pathlib import Path
from urllib.request import urlretrieve

from huggingface_hub import hf_hub_download


REPO_ID = "vidit031/isl-isolated-40words"
REPO_TYPE = "dataset"

ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "data" / "dataset"


TARGET_LABELS = {
    "hello",
    "thank you",
    "please",
    "yes",
    "no",
    "help",
    "stop",
    "water",
    "food",
    "goodbye",
}


def normalize_label(label: str) -> str:
    return (
        label.strip()
        .lower()
        .replace("_", " ")
    )


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("======================================")
    print(" Downloading ISL dataset from HuggingFace")
    print("======================================")
    print()

    metadata_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="metadata.csv",
        repo_type=REPO_TYPE,
    )

    print(f"[OK] Metadata downloaded:")
    print(metadata_path)

    selected_rows = []

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
        newline="",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            label = normalize_label(
                row["word"]
            )

            if label in TARGET_LABELS:
                selected_rows.append(row)

    print()
    print(
        f"[OK] Selected clips: "
        f"{len(selected_rows)}"
    )

    for index, row in enumerate(
        selected_rows,
        start=1,
    ):

        label = normalize_label(
            row["word"]
        )

        video_path = row["video_path"]

        safe_label = (
            label
            .replace(" ", "_")
        )

        output_label_dir = (
            OUTPUT_DIR / safe_label
        )

        output_label_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            output_label_dir
            / Path(video_path).name
        )

        if output_file.exists():
            print(
                f"[SKIP] "
                f"{index}/{len(selected_rows)} "
                f"{label}"
            )
            continue

        print(
            f"[DOWNLOAD] "
            f"{index}/{len(selected_rows)} "
            f"{label}"
        )

        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=video_path,
                repo_type=REPO_TYPE,
            )

            output_file.write_bytes(
                Path(downloaded_path).read_bytes()
            )

        except Exception as error:
            print(
                f"[ERROR] "
                f"{label}: {error}"
            )

    print()
    print("[DONE]")
    print()
    print(
        f"Dataset location:\n"
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()