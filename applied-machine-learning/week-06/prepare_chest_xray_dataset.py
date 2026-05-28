#!/usr/bin/env python3
"""Prepare the 3-class balanced chest X-ray dataset for Week 6.

Transforms the original Kaggle/Mendeley Chest X-Ray (Pneumonia) dataset into the
layout expected by Week6_Pneumonia_Assignment.ipynb:

    chest_xray/
    ├── train/{NORMAL,BACTERIAL_PNEUMONIA,VIRAL_PNEUMONIA}/
    └── val/{NORMAL,BACTERIAL_PNEUMONIA,VIRAL_PNEUMONIA}/
    unseen/{NORMAL,BACTERIAL_PNEUMONIA,VIRAL_PNEUMONIA}/

Steps:
  1. Split PNEUMONIA into bacterial / viral using filename patterns.
  2. Merge original test + val splits into a single validation set.
  3. Hold out fixed unseen images (section 5.3 of the assignment).
  4. Balance each split by subsampling to the smallest class count.

Source dataset:
  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
  https://data.mendeley.com/datasets/rscbjbr9sj/3
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import shutil
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

CLASS_NAMES = ("NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")
ORIGINAL_SPLITS = ("train", "test", "val")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}

# Holdout images referenced in Week6_Pneumonia_Assignment.ipynb (section 5.3).
UNSEEN_FILES = {
    "BACTERIAL_PNEUMONIA/person296_bacteria_1394.jpeg",
    "BACTERIAL_PNEUMONIA/person441_bacteria_1916.jpeg",
    "BACTERIAL_PNEUMONIA/person564_bacteria_2342.jpeg",
    "NORMAL/IM-0353-0001.jpeg",
    "NORMAL/IM-0633-0001.jpeg",
    "NORMAL/NORMAL2-IM-0866-0001.jpeg",
    "VIRAL_PNEUMONIA/person1369_virus_2356.jpeg",
    "VIRAL_PNEUMONIA/person1465_virus_2537.jpeg",
    "VIRAL_PNEUMONIA/person1537_virus_2674.jpeg",
}

KAGGLE_DATASET = "paultimothymooney/chest-xray-pneumonia"
KAGGLE_DOWNLOAD_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia"
)
HF_DATASET_REPO = "opencampus/chest-xray-pneumonia-3class-balanced"
PREPARED_ZIP_NAME = "chest_xray_prepared.zip"
WEIGHTS_FILENAME = "resnet18_chest_xray_classifier_weights.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory that will contain chest_xray/ and unseen/ (default: current dir).",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Path to extracted original dataset root (contains train/, test/, val/).",
    )
    parser.add_argument(
        "--download-kaggle",
        action="store_true",
        help="Download the original dataset from Kaggle (requires KAGGLE_USERNAME/KAGGLE_KEY).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
        help="Directory for downloaded archives (default: ./.cache).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when subsampling for class balance (default: 42).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing prepared output directories.",
    )
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Upload prepared folders to Hugging Face Hub after preparation.",
    )
    parser.add_argument(
        "--colab-assets-only",
        action="store_true",
        help="With --upload-hf, upload only chest_xray_prepared.zip and weights (skip image folders).",
    )
    parser.add_argument(
        "--hf-repo-id",
        default="opencampus/chest-xray-pneumonia-3class-balanced",
        help="Hugging Face dataset repo id (default: opencampus/chest-xray-pneumonia-3class-balanced).",
    )
    return parser.parse_args()


def classify_filename(filename: str) -> str:
    name = filename.lower()
    if "_bacteria_" in name:
        return "BACTERIAL_PNEUMONIA"
    if "_virus_" in name:
        return "VIRAL_PNEUMONIA"
    raise ValueError(f"Cannot infer pneumonia subtype from filename: {filename}")


def iter_images(class_dir: Path) -> list[Path]:
    if not class_dir.is_dir():
        return []
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def collect_original(source_root: Path) -> dict[str, dict[str, list[Path]]]:
    """Return {split: {class_name: [paths]}} with 3-class labels."""
    inventory: dict[str, dict[str, list[Path]]] = {
        split: defaultdict(list) for split in ORIGINAL_SPLITS
    }

    for split in ORIGINAL_SPLITS:
        split_dir = source_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split_dir}")

        inventory[split]["NORMAL"].extend(iter_images(split_dir / "NORMAL"))

        pneumonia_dir = split_dir / "PNEUMONIA"
        for image_path in iter_images(pneumonia_dir):
            class_name = classify_filename(image_path.name)
            inventory[split][class_name].append(image_path)

    return inventory


def rel_class_path(class_name: str, filename: str) -> str:
    return f"{class_name}/{filename}"


def reserve_unseen(
    pools: dict[str, list[Path]],
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Move unseen holdout images out of train/val pools into unseen/."""
    unseen: dict[str, list[Path]] = defaultdict(list)
    remaining: dict[str, list[Path]] = defaultdict(list)

    path_lookup: dict[str, Path] = {}
    for class_name, paths in pools.items():
        for path in paths:
            key = rel_class_path(class_name, path.name)
            if key in path_lookup:
                raise RuntimeError(f"Duplicate filename across classes: {key}")
            path_lookup[key] = path

    missing = [name for name in UNSEEN_FILES if name not in path_lookup]
    if missing:
        raise FileNotFoundError(
            "Could not locate unseen holdout images in source data:\n"
            + "\n".join(f"  - {name}" for name in missing)
        )

    unseen_keys = set(UNSEEN_FILES)
    for class_name, paths in pools.items():
        for path in paths:
            key = rel_class_path(class_name, path.name)
            if key in unseen_keys:
                unseen[class_name].append(path)
            else:
                remaining[class_name].append(path)

    return remaining, unseen


def balance(
    pools: dict[str, list[Path]], seed: int
) -> dict[str, list[Path]]:
    counts = {class_name: len(paths) for class_name, paths in pools.items()}
    if not counts or min(counts.values()) == 0:
        raise ValueError(f"Cannot balance empty class pool: {counts}")

    target = min(counts.values())
    rng = random.Random(seed)
    balanced: dict[str, list[Path]] = {}
    for class_name, paths in pools.items():
        if len(paths) < target:
            raise ValueError(
                f"Class {class_name} has only {len(paths)} images, "
                f"but target is {target}."
            )
        balanced[class_name] = sorted(rng.sample(paths, target))
    return balanced


def copy_tree(
    class_to_paths: dict[str, list[Path]],
    destination_root: Path,
    force: bool,
) -> None:
    if destination_root.exists():
        if not force:
            raise FileExistsError(
                f"Output directory already exists: {destination_root}. "
                "Use --force to overwrite."
            )
        shutil.rmtree(destination_root)

    for class_name in CLASS_NAMES:
        (destination_root / class_name).mkdir(parents=True, exist_ok=True)

    for class_name, paths in class_to_paths.items():
        for src in paths:
            dst = destination_root / class_name / src.name
            shutil.copy2(src, dst)


def summarize(label: str, class_to_paths: dict[str, list[Path]]) -> dict[str, int]:
    counts = {class_name: len(class_to_paths.get(class_name, [])) for class_name in CLASS_NAMES}
    total = sum(counts.values())
    print(f"\n{label}")
    for class_name in CLASS_NAMES:
        print(f"  - {class_name:<22}: {counts[class_name]:>5} images")
    print(f"  {'Total':<22}: {total:>5} images")
    return counts


def _load_kaggle_credentials() -> tuple[str, str]:
    username = os.environ.get("KAGGLE_USERNAME")
    api_key = os.environ.get("KAGGLE_KEY")
    if username and api_key:
        return username, api_key

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.is_file():
        data = json.loads(kaggle_json.read_text(encoding="utf-8"))
        username = data.get("username")
        api_key = data.get("key")
        if username and api_key:
            return username, api_key

    raise EnvironmentError(
        "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY, "
        "or create ~/.kaggle/kaggle.json, "
        "or pass --source-dir pointing to an extracted chest_xray folder."
    )


def download_kaggle_dataset(cache_dir: Path) -> Path:
    username, api_key = _load_kaggle_credentials()

    cache_dir.mkdir(parents=True, exist_ok=True)
    zip_path = cache_dir / "chest-xray-pneumonia.zip"
    extract_dir = cache_dir / "kaggle_chest_xray"

    if not zip_path.exists():
        print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
        token = base64.b64encode(f"{username}:{api_key}".encode()).decode("ascii")
        request = Request(
            KAGGLE_DOWNLOAD_URL,
            headers={
                "User-Agent": "opencampus-week-06-dataset-prep",
                "Authorization": f"Basic {token}",
            },
        )
        try:
            with urlopen(request, timeout=120) as response:
                if response.status != 200:
                    raise RuntimeError(f"Unexpected Kaggle response status: {response.status}")
                zip_path.write_bytes(response.read())
        except HTTPError as exc:
            raise RuntimeError(
                "Kaggle download failed. Verify KAGGLE_USERNAME/KAGGLE_KEY and dataset access."
            ) from exc
        except URLError as exc:
            raise RuntimeError("Network error while downloading from Kaggle.") from exc
        print(f"Saved archive to {zip_path}")

    if not extract_dir.exists():
        print(f"Extracting {zip_path}...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(extract_dir)

    candidates = [extract_dir, extract_dir / "chest_xray"]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate

    raise FileNotFoundError(
        f"Could not locate train/test folders under extracted archive: {extract_dir}"
    )


def resolve_source_dir(args: argparse.Namespace) -> Path:
    if args.source_dir is not None:
        source = args.source_dir.expanduser().resolve()
        if not (source / "train").is_dir():
            raise FileNotFoundError(f"--source-dir must contain train/: {source}")
        return source

    if args.download_kaggle:
        return download_kaggle_dataset(args.cache_dir.resolve())

    default_cache = (args.cache_dir / "kaggle_chest_xray" / "chest_xray").resolve()
    if default_cache.is_dir():
        print(f"Using cached source dataset: {default_cache}")
        return default_cache

    raise SystemExit(
        "No source dataset found. Either:\n"
        "  1) pass --download-kaggle with KAGGLE_USERNAME and KAGGLE_KEY set, or\n"
        "  2) pass --source-dir /path/to/extracted/chest_xray"
    )


def create_prepared_zip(output_dir: Path, zip_path: Path | None = None) -> Path:
    """Pack chest_xray/ and unseen/ into a single zip for Colab wget downloads."""
    upload_root = output_dir.resolve()
    chest_xray_dir = upload_root / "chest_xray"
    unseen_dir = upload_root / "unseen"
    if not chest_xray_dir.is_dir() or not unseen_dir.is_dir():
        raise FileNotFoundError("Expected chest_xray/ and unseen/ under output directory.")

    zip_path = (zip_path or upload_root / PREPARED_ZIP_NAME).resolve()
    if zip_path.exists():
        zip_path.unlink()

    print(f"Creating {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as archive:
        for folder_name in ("chest_xray", "unseen"):
            folder = upload_root / folder_name
            for file_path in sorted(folder.rglob("*")):
                if file_path.is_file():
                    archive.write(file_path, file_path.relative_to(upload_root).as_posix())

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {zip_path} ({size_mb:.1f} MB)")
    return zip_path


def upload_to_huggingface(
    output_dir: Path,
    repo_id: str,
    weights_path: Path | None = None,
    upload_folders: bool = True,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError("Install huggingface_hub to use --upload-hf.") from exc

    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True)

    metadata = {
        "source": KAGGLE_DATASET,
        "mendeley_doi": "10.17632/rscbjbr9sj.3",
        "classes": list(CLASS_NAMES),
        "preparation_script": "prepare_chest_xray_dataset.py",
    }
    readme = f"""---
license: cc-by-4.0
task_categories:
- image-classification
tags:
- medical
- chest-xray
- pneumonia
- opencampus
size_categories:
- 1K<n<10K
---

# Chest X-Ray (Pneumonia) — 3-class balanced (opencampus Week 6)

Prepared dataset for the opencampus applied ML Week 6 assignment.

## Classes

- `NORMAL`
- `BACTERIAL_PNEUMONIA`
- `VIRAL_PNEUMONIA`

## Layout

```
chest_xray/
├── train/
└── val/
unseen/
```

## Colab / notebook download

For Google Colab, download the prepared archive (~1 GB) and pretrained weights from this repo:

- `chest_xray_prepared.zip` — contains `chest_xray/` and `unseen/`
- `{WEIGHTS_FILENAME}` — pretrained ResNet-18 weights for section 5

Download `xray_viz.py` from the [course-material](https://github.com/opencampus-sh/course-material) GitHub repository.

## Source

Derived from the [Chest X-Ray Images (Pneumonia)]({KAGGLE_DOWNLOAD_URL.replace('/api/v1/datasets/download/', '/datasets/')}) dataset
(Kermany et al., Mendeley DOI [10.17632/rscbjbr9sj.3](https://doi.org/10.17632/rscbjbr9sj.3)).

## Preparation

See `prepare_chest_xray_dataset.py` in the course repository for the exact steps.

```json
{json.dumps(metadata, indent=2)}
```
"""

    upload_root = output_dir.resolve()
    chest_xray_dir = upload_root / "chest_xray"
    unseen_dir = upload_root / "unseen"
    if not chest_xray_dir.is_dir() or not unseen_dir.is_dir():
        raise FileNotFoundError("Expected chest_xray/ and unseen/ under output directory.")

    zip_path = create_prepared_zip(upload_root)
    weights_path = (weights_path or Path(__file__).resolve().parent / WEIGHTS_FILENAME).resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

    print(f"\nUploading dataset to Hugging Face: {repo_id}")
    if upload_folders:
        api.upload_folder(
            folder_path=str(chest_xray_dir),
            path_in_repo="chest_xray",
            repo_id=repo_id,
            repo_type="dataset",
        )
        api.upload_folder(
            folder_path=str(unseen_dir),
            path_in_repo="unseen",
            repo_id=repo_id,
            repo_type="dataset",
        )

    api.upload_file(
        path_or_fileobj=str(zip_path),
        path_in_repo=PREPARED_ZIP_NAME,
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(weights_path),
        path_in_repo=WEIGHTS_FILENAME,
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"Upload complete: https://huggingface.co/datasets/{repo_id}")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    source_root = resolve_source_dir(args)

    print(f"Source dataset: {source_root}")
    print(f"Output directory: {output_dir}")

    inventory = collect_original(source_root)

    train_pool: dict[str, list[Path]] = {
        class_name: list(inventory["train"][class_name]) for class_name in CLASS_NAMES
    }
    val_pool: dict[str, list[Path]] = {
        class_name: inventory["test"][class_name] + inventory["val"][class_name]
        for class_name in CLASS_NAMES
    }

    combined = {
        class_name: train_pool[class_name] + val_pool[class_name]
        for class_name in CLASS_NAMES
    }
    _, unseen = reserve_unseen(combined)

    train_without_unseen = {
        class_name: [
            path
            for path in train_pool[class_name]
            if rel_class_path(class_name, path.name)
            not in {rel for rel in UNSEEN_FILES}
        ]
        for class_name in CLASS_NAMES
    }
    val_without_unseen = {
        class_name: [
            path
            for path in val_pool[class_name]
            if rel_class_path(class_name, path.name)
            not in {rel for rel in UNSEEN_FILES}
        ]
        for class_name in CLASS_NAMES
    }

    train_balanced = balance(train_without_unseen, seed=args.seed)
    val_balanced = balance(val_without_unseen, seed=args.seed + 1)

    chest_xray_root = output_dir / "chest_xray"
    unseen_root = output_dir / "unseen"

    copy_tree(train_balanced, chest_xray_root / "train", force=args.force)
    copy_tree(val_balanced, chest_xray_root / "val", force=args.force)
    copy_tree(unseen, unseen_root, force=True)

    train_counts = summarize("Prepared training set", train_balanced)
    val_counts = summarize("Prepared validation set", val_balanced)
    unseen_counts = summarize("Unseen holdout set", unseen)

    manifest = {
        "source_root": str(source_root),
        "seed": args.seed,
        "train": train_counts,
        "val": val_counts,
        "unseen": unseen_counts,
        "unseen_files": sorted(UNSEEN_FILES),
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote manifest: {manifest_path}")

    if args.upload_hf:
        upload_to_huggingface(
            output_dir,
            args.hf_repo_id,
            upload_folders=not args.colab_assets_only,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
