"""Visualization and evaluation helpers for the Week 6 chest X-ray assignment."""

from __future__ import annotations

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from torchmetrics.classification import MulticlassConfusionMatrix
from torchvision import transforms
from tqdm.auto import tqdm

CLASS_ORDER = ("NORMAL", "BACTERIAL_PNEUMONIA", "VIRAL_PNEUMONIA")
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}
NORMALIZE_MEAN = (0.482, 0.482, 0.482)
NORMALIZE_STD = (0.222, 0.222, 0.222)


def count_images_by_class(dataset_root: str | Path) -> None:
    """Print per-class image counts for the train and val folders."""
    root = Path(dataset_root)
    name_width = 22

    for split in ("train", "val"):
        split_dir = root / split
        print(f"--- {split.title()} ---")
        if not split_dir.is_dir():
            print(f"  Missing folder: {split_dir}\n")
            continue

        total = 0
        for class_name in sorted(
            d.name for d in split_dir.iterdir() if d.is_dir()
        ):
            count = sum(
                1
                for path in (split_dir / class_name).iterdir()
                if path.suffix.lower() in IMAGE_SUFFIXES
            )
            total += count
            print(f"  {class_name:<{name_width}} {count:>5}")

        print(f"  {'Total':<{name_width}} {total:>5}\n")


def show_training_samples(train_dir: str | Path, seed: int = 0) -> None:
    """Show two random training images per class."""
    rng = random.Random(seed)
    train_dir = Path(train_dir)
    fig, axes = plt.subplots(len(CLASS_ORDER), 2, figsize=(7, 7))

    for row, class_name in enumerate(CLASS_ORDER):
        class_dir = train_dir / class_name
        if not class_dir.is_dir():
            axes[row, 0].axis("off")
            axes[row, 1].axis("off")
            continue

        files = [
            path
            for path in class_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        ]
        if len(files) < 2:
            print(f"Not enough images in {class_name}")
            continue

        for col, image_path in enumerate(rng.sample(files, 2)):
            axes[row, col].imshow(Image.open(image_path), cmap="gray")
            axes[row, col].set_title(class_name, fontsize=11)
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def _plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="g",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix")
    plt.show()


def validation_report(model, data_module) -> None:
    """Run inference on the validation set and show per-class accuracy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    preds, labels = [], []
    loader = tqdm(data_module.val_dataloader(), desc="Validation", leave=False)

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(images)
            preds.append(torch.argmax(outputs, dim=1))
            labels.append(batch_labels)

    preds = torch.cat(preds)
    labels = torch.cat(labels)

    num_classes = model.hparams.num_classes
    cm = MulticlassConfusionMatrix(num_classes=num_classes).to(device)(preds, labels)
    per_class_acc = cm.diag() / cm.sum(dim=1)
    class_names = data_module.val_dataset.classes

    print("Per-class accuracy")
    for name, acc in zip(class_names, per_class_acc):
        print(f"  {name:<22} {acc.item():.4f}")

    _plot_confusion_matrix(cm.cpu().numpy(), class_names)


def show_validation_predictions(model, data_module, seed: int = 1) -> None:
    """Plot random validation images with true and predicted labels."""
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    val_dataset = data_module.val_dataset
    class_names = val_dataset.classes
    targets = np.array(val_dataset.targets)

    picked_indices: list[int] = []
    for class_id in range(len(class_names)):
        class_indices = np.where(targets == class_id)[0]
        if len(class_indices) >= 2:
            picked_indices.extend(rng.sample(class_indices.tolist(), 2))

    if not picked_indices:
        print("Could not sample validation images.")
        return

    images = torch.stack([val_dataset[i][0] for i in picked_indices]).to(device)
    true_labels = torch.tensor([val_dataset[i][1] for i in picked_indices])

    with torch.no_grad():
        pred_labels = torch.argmax(model(images), dim=1)

    fig, axes = plt.subplots(len(class_names), 2, figsize=(8, 8))
    fig.suptitle("Validation predictions", fontsize=15)

    mean = np.array(NORMALIZE_MEAN)
    std = np.array(NORMALIZE_STD)

    for plot_idx, dataset_idx in enumerate(picked_indices):
        ax = axes[plot_idx // 2, plot_idx % 2]
        image = val_dataset[dataset_idx][0].permute(1, 2, 0).numpy()
        image = np.clip(std * image + mean, 0, 1)
        ax.imshow(image)
        ax.axis("off")

        true_name = class_names[true_labels[plot_idx]]
        pred_name = class_names[pred_labels[plot_idx]]
        color = "green" if true_name == pred_name else "red"
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", color=color, fontsize=10)

    plt.tight_layout()
    plt.show()


def predict_and_show(model, data_module, image_path: str | Path) -> None:
    """Load one image from disk and display the model prediction."""
    image_path = Path(image_path)
    class_names = data_module.train_dataset.classes

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ])

    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Image not found: {image_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    batch = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = torch.argmax(model(batch), dim=1).item()

    plt.imshow(image, cmap="gray")
    plt.title(f"Prediction: {class_names[prediction]}")
    plt.axis("off")
    plt.show()
