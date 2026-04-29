import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.optim as optim
import torchmetrics
from IPython.display import Markdown, display
from torch.utils.data import random_split
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def load_bert(model_id="distilbert-base-uncased", num_classes=4):
    """
    Downloads (if needed) and loads a model from the HuggingFace Hub using the
    safe safetensors format, then attaches a fresh classification head.

    The safetensors format is preferred over the legacy pytorch_model.bin because
    it cannot execute arbitrary code on load (unlike Python pickle).

    Args:
        model_id: HuggingFace Hub model identifier.
        num_classes: Number of output classes for the classification head.
                     Defaults to 4 for the AG News topic classification task.

    Returns:
        A tuple containing the loaded model (with the new head) and the tokenizer.
    """
    print(f"Loading '{model_id}' from the HuggingFace Hub (safetensors format)...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_classes,
        use_safetensors=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully ({total_params:,} parameters).")

    return model, tokenizer


def create_dataset_splits(full_dataset, train_split_percentage=0.8):
    """
    Splits a full dataset into training and validation sets.

    Args:
        full_dataset: The complete PyTorch Dataset to be split.
        train_split_percentage: The percentage of the dataset to allocate
                                for the training set. Defaults to 0.8.

    Returns:
        A tuple containing the training dataset and validation dataset.
    """
    train_size = int(train_split_percentage * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    return train_dataset, val_dataset


def training_loop(
    model, train_loader, val_loader, loss_function, learning_rate, num_epochs, device
):
    """
    Performs a full training and validation cycle for a PyTorch model.

    Args:
        model: The PyTorch model to be trained.
        train_loader: The DataLoader for the training dataset.
        val_loader: The DataLoader for the validation dataset.
        loss_function: The loss function used for training.
        learning_rate: The learning rate for the optimizer.
        num_epochs: The total number of epochs to train for.
        device: The computational device ('cuda' or 'cpu') to run on.

    Returns:
        A tuple containing the trained model and a dictionary of the final
        performance metrics from the last validation epoch:
            - "val_loss": float
            - "val_accuracy": float
            - "val_f1": float
            - "confusion_matrix": CPU tensor of shape (num_classes, num_classes)
    """
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    num_classes = model.config.num_labels

    val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(
        device
    )
    val_f1 = torchmetrics.F1Score(
        task="multiclass", num_classes=num_classes, average="macro"
    ).to(device)
    val_cm = torchmetrics.ConfusionMatrix(
        task="multiclass", num_classes=num_classes
    ).to(device)

    epoch_loop = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in epoch_loop:

        # --- Training Phase ---
        model.train()
        train_loss_epoch = 0

        train_inner_loop = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False
        )
        for batch in train_inner_loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_function(logits, labels)

            train_loss_epoch += loss.item()
            loss.backward()

            optimizer.step()

            train_inner_loop.set_postfix(loss=loss.item())

        train_loss_epoch /= len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss_epoch = 0

        val_inner_loop = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False
        )
        with torch.no_grad():
            for batch in val_inner_loop:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                val_loss = loss_function(logits, labels)
                val_loss_epoch += val_loss.item()

                preds = torch.argmax(logits, dim=-1)
                val_accuracy.update(preds, labels)
                val_f1.update(preds, labels)
                val_cm.update(preds, labels)

        val_loss_epoch /= len(val_loader)

        epoch_acc = val_accuracy.compute()
        epoch_f1 = val_f1.compute()

        val_accuracy.reset()
        val_f1.reset()
        val_cm.reset()

        epoch_loop.set_postfix(
            train_loss=f"{train_loss_epoch:.4f}",
            val_loss=f"{val_loss_epoch:.4f}",
            val_acc=f"{epoch_acc:.4f}",
        )
        tqdm.write(
            f"Epoch {epoch+1} Metrics -> Val Loss: {val_loss_epoch:.4f}, "
            f"Val Acc: {epoch_acc:.4f}, Val F1: {epoch_f1:.4f}"
        )

    print("\n--- Training complete ---")

    # Run one final validation pass to get the confusion matrix for the trained model
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            val_cm.update(preds, labels)

    final_confusion_matrix = val_cm.compute()

    final_results = {
        "val_loss": val_loss_epoch,
        "val_accuracy": epoch_acc.item(),
        "val_f1": epoch_f1.item(),
        "confusion_matrix": final_confusion_matrix.cpu(),
    }

    return model, final_results


def _create_accuracy_table(class_accuracy, id2cat):
    """
    Formats per-class accuracy data into a markdown table string.

    Args:
        class_accuracy (np.array): An array of accuracy values for each class.
        id2cat (dict): A dictionary mapping class IDs to category names.

    Returns:
        str: A formatted markdown string representing the accuracy table.
    """
    markdown_table = "| Category                  | Accuracy |\n"
    markdown_table += "|---------------------------|----------|\n"

    for class_id, accuracy in enumerate(class_accuracy):
        category_name = id2cat[class_id]
        markdown_table += f"| {category_name:<25} | {accuracy:.2%}    |\n"

    return markdown_table


def analyze_and_plot_results(results, id2cat):
    """
    Analyzes training results to display per-class accuracy and plot a
    confusion matrix.

    Args:
        results (dict): The results dictionary from the training loop, expected
                        to contain a 'confusion_matrix' tensor.
        id2cat (dict): A dictionary mapping class IDs to category names.
    """
    confusion_matrix_numpy = results["confusion_matrix"].cpu().numpy()

    class_names = [name for id, name in sorted(id2cat.items())]

    correct_predictions = confusion_matrix_numpy.diagonal()
    total_samples_per_class = confusion_matrix_numpy.sum(axis=1)
    class_accuracy = correct_predictions / (total_samples_per_class + 1e-9)

    accuracy_table_md = _create_accuracy_table(class_accuracy, id2cat)

    display(Markdown("### **Per-Class Accuracy**"))
    display(Markdown(accuracy_table_md))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix_numpy,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.title("Confusion Matrix — AG News Topic Classifier")
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.show()


def predict_category(model, tokenizer, text, device, id2cat):
    """
    Performs inference on a single text string to predict its news category.

    Args:
        model (nn.Module): The fine-tuned PyTorch model.
        tokenizer: The Hugging Face tokenizer corresponding to the model.
        text (str): The raw input news article or headline.
        device: The device to perform inference on ('cuda', 'cpu', etc.).
        id2cat (dict): A dictionary mapping class IDs to category names.

    Returns:
        str: The predicted category name as a string.
    """
    model.eval()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_id = torch.argmax(logits, dim=-1).item()

    return id2cat[predicted_id]
