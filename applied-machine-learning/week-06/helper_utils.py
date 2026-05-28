import os
import random

import numpy as np
import seaborn as sns
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm

from torchvision import transforms
from torchmetrics.classification import MulticlassConfusionMatrix
import torch.nn as nn
import torchvision.models as tv_models



def display_dataset_count(dataset_path):
    """
    Calculates and displays the number of images for each class and the total
    number of images in the train and validation sets.

    Args:
        dataset_path: The root path to the dataset directory.
    """
    # Define a maximum width for class names and the total label to ensure alignment
    class_width = 25
    total_width = 25

    # Iterate over the main directories (train, val)
    for split in ['train', 'val']:
        # Construct the full path for the current split
        split_path = os.path.join(dataset_path, split)
        
        # Check if the directory exists
        if not os.path.isdir(split_path):
            # Print a message if the directory is not found and continue to the next split
            print(f"\nDirectory not found: {split_path}")
            continue
            
        # Print a header for the current dataset split
        print(f"--- {split.capitalize()} Set ---")
        
        # Initialize a counter for the total number of images in the current split
        total_images = 0
        
        # Get class names from the subdirectories within the split path and sort them
        class_names = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        
        # Check if any class directories were found
        if not class_names:
            # Print a message if no class directories are found and continue to the next split
            print("  No class directories found.")
            continue

        # Count images in each class directory and add to total
        for class_name in class_names:
            # Construct the full path for the current class directory
            class_path = os.path.join(split_path, class_name)
            # Count the number of items (images) in the class directory
            num_images = len(os.listdir(class_path))
            # Add the number of images in the current class to the total count
            total_images += num_images
            # Use f-string with left-alignment for class name and right-alignment for image count
            print(f"  - {class_name:<{class_width}}: {num_images} images")
            
        # Print a separator line for readability
        print("—" * (class_width + total_width))
        # Print the total count for the current set with right-alignment
        print(f"  Total: {total_images:>{total_width}} images\n")
        
        

def display_random_images(path):
    """
    Displays a grid of random images from the dataset.
    
    This function shows two random images for each class, ensuring the rows
    always follow the order: NORMAL, BACTERIAL_PNEUMONIA, VIRAL_PNEUMONIA.

    Args:
        path: The path to the training data directory (e.g., './chest_xray/train').
    """
    try:
        # Define the exact order for class display
        class_names = ['NORMAL', 'BACTERIAL_PNEUMONIA', 'VIRAL_PNEUMONIA']
        # Get the number of classes
        num_classes = len(class_names)
        
        # Create a figure with a row for each class and two columns for images
        fig, axs = plt.subplots(num_classes, 2, figsize=(7, 7))

        # Loop through each class in the predefined order
        for i, class_name in enumerate(class_names):
            # Construct the path to the current class directory
            class_path = os.path.join(path, class_name)
            
            # Check if the class directory actually exists
            if not os.path.isdir(class_path):
                # Print a warning and turn off the axes for the missing directory
                print(f"Warning: Directory for class '{class_name}' not found. Skipping row.")
                # Turn off the axis for the first subplot in the current row
                axs[i, 0].axis('off')
                # Turn off the axis for the second subplot in the current row
                axs[i, 1].axis('off')
                # Continue to the next iteration of the loop
                continue
            
            # Get a list of all image files in the class directory
            image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Check if there are at least two images in the class directory
            if len(image_files) < 2:
                # Print a warning and turn off the axes if there are not enough images
                print(f"Warning: Class '{class_name}' has fewer than 2 images. Skipping.")
                # Turn off the axis for the first subplot in the current row
                axs[i, 0].axis('off')
                # Turn off the axis for the second subplot in the current row
                axs[i, 1].axis('off')
                # Continue to the next iteration of the loop
                continue

            # Randomly sample two unique image files
            selected_images = random.sample(image_files, 2)
            
            # Display the two selected images
            for j, image_file in enumerate(selected_images):
                # Construct the full path to the image file
                image_path = os.path.join(class_path, image_file)
                # Open the image using the PIL library
                image = Image.open(image_path)
                
                # Select the appropriate subplot for the current image
                ax = axs[i, j]
                # Display the image on the subplot with a grayscale colormap
                ax.imshow(image, cmap='gray')
                # Turn off the axis to hide ticks and labels
                ax.axis('off')
                # Set the title of the subplot to the class name
                ax.set_title(class_name, fontsize=12)

        # Adjust the subplot parameters for a tight layout
        plt.tight_layout()
        # Display the figure with the images
        plt.show()

    except FileNotFoundError:
        # Handle the case where the main directory is not found
        print(f"Error: The directory '{path}' was not found.")
    except Exception as e:
        # Handle any other unexpected exceptions
        print(f"An unexpected error occurred: {e}")
        
        
        
def plot_confusion_matrix(cm, class_names):
    """
    Plots a confusion matrix using Seaborn's heatmap.

    Args:
        cm: A confusion matrix numpy array.
        class_names: A list of class names for the labels.
    """
    # Create a new figure with a specified size
    plt.figure(figsize=(8, 6))
    # Generate a heatmap from the confusion matrix data
    sns.heatmap(
        cm, 
        # Annotate cells with the numeric value
        annot=True,     
        # Use a general format to disable scientific notation
        fmt='g',        
        # Set the color map to 'Blues'
        cmap='Blues',   
        # Set the labels for the x-axis ticks
        xticklabels=class_names, 
        # Set the labels for the y-axis ticks
        yticklabels=class_names
    )
    # Set the label for the x-axis
    plt.xlabel('Predicted Labels')
    # Set the label for the y-axis
    plt.ylabel('True Labels')
    # Set the title of the plot
    plt.title('Confusion Matrix')
    # Display the plot
    plt.show()
    
    
    
def per_class_acc_and_conf_matrix(trained_model, data_module):
    """
    Evaluates a trained model on a validation dataset and prints a per-class
    accuracy report and confusion matrix.

    Args:
        trained_model: The trained Lightning model to evaluate.
        data_module: The data module containing the validation dataloader.
    """
    # --- Setup ---
    # Set the model to evaluation mode
    trained_model.eval()
    # Determine the device to use for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model to the selected device
    trained_model = trained_model.to(device)

    # Initialize lists to store predictions and true labels
    all_preds = []
    all_labels = []

    # --- Run Inference ---
    # Wrap the dataloader with tqdm for a progress bar
    val_loader_with_progress = tqdm(
        data_module.val_dataloader(), 
        # Set a description for the progress bar
        desc="Evaluating Model", 
        # Do not leave the progress bar after completion
        leave=False
    )

    # Disable gradient calculations for inference
    with torch.no_grad():
        # Iterate over batches in the validation dataloader
        for batch in val_loader_with_progress:
            # Unpack images and labels from the batch
            images, labels = batch
            # Move images and labels to the appropriate device
            images, labels = images.to(device), labels.to(device)
            
            # Perform a forward pass to get model outputs
            outputs = trained_model(images)
            # Get the predicted class by finding the index with the highest value
            preds = torch.argmax(outputs, dim=1)
            
            # Append the predictions from the current batch to the list
            all_preds.append(preds)
            # Append the true labels from the current batch to the list
            all_labels.append(labels)

    # --- Calculate and Display Metrics ---
    # Concatenate all predictions into a single tensor
    all_preds = torch.cat(all_preds)
    # Concatenate all true labels into a single tensor
    all_labels = torch.cat(all_labels)

    # Initialize the confusion matrix metric
    confmat = MulticlassConfusionMatrix(num_classes=trained_model.hparams.num_classes).to(device)
    # Compute the confusion matrix
    cm = confmat(all_preds, all_labels)

    # Calculate per-class accuracy from the confusion matrix
    per_class_acc = cm.diag() / cm.sum(axis=1)
    # Get the names of the classes from the dataset
    class_names = data_module.val_dataset.classes

    # Print a header for the accuracy report
    print("--- Per-Class Accuracy Report ---")
    # Iterate through each class and print its accuracy
    for i, acc in enumerate(per_class_acc):
        # Print the accuracy for the current class
        print(f"  - Accuracy for class '{class_names[i]}': {acc.item():.4f}")
    # Print a new line for spacing
    print()

    # Plot the confusion matrix
    plot_confusion_matrix(cm.cpu().numpy(), class_names)
    
    
    
def display_random_predictions(trained_model, data_module):
    """
    Displays two random validation images from each class along with their
    true and predicted labels.

    Args:
        trained_model: The trained model to use for predictions.
        data_module: The data module containing the validation dataset.
    """
    # --- Setup ---
    # Set the model to evaluation mode
    trained_model.eval()
    # Determine the device to use for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model to the selected device
    trained_model = trained_model.to(device)
    
    # Get the validation dataset from the data module
    val_dataset = data_module.val_dataset
    # Get the class names from the dataset
    class_names = val_dataset.classes
    # Get the number of classes
    num_classes = len(class_names)

    # Get indices for each class in the validation set
    class_indices = {i: np.where(np.array(val_dataset.targets) == i)[0] 
                      for i in range(num_classes)}

    # --- Randomly Select Images ---
    # Initialize a list to store the indices of the selected images
    selected_indices = []
    # Iterate through each class
    for i in range(num_classes):
        # Check if there are at least two images for the current class
        if len(class_indices[i]) >= 2:
            # Randomly select two indices and add them to the list
            selected_indices.extend(random.sample(list(class_indices[i]), 2))
        else:
            # Print a warning if there are not enough images
            print(f"Warning: Class '{class_names[i]}' has fewer than 2 validation images.")

    # Check if any images were selected for display
    if not selected_indices:
        # Print a message if no images could be found
        print("Could not find enough images to display.")
        # Exit the function
        return

    # --- Get Predictions ---
    # Stack the selected images into a single tensor and move to the device
    images = torch.stack([val_dataset[i][0] for i in selected_indices]).to(device)
    # Create a tensor of the corresponding true labels
    labels = torch.tensor([val_dataset[i][1] for i in selected_indices])

    # Disable gradient calculations
    with torch.no_grad():
        # Get model outputs for the selected images
        outputs = trained_model(images)
        # Get the predicted class by finding the index of the maximum value
        preds = torch.argmax(outputs, dim=1)

    # --- Display Results ---
    # Adjust figure size for better aspect ratio
    fig, axs = plt.subplots(num_classes, 2, figsize=(8, 8))
    # Set the main title of the figure
    fig.suptitle('Validation Image Predictions', fontsize=16, x=0.4)

    # Iterate through each selected image and its index
    for i, idx in enumerate(selected_indices):
        # Select the correct subplot based on the index
        ax = axs[i // 2, i % 2]
        
        # Un-normalize and display the image
        img = val_dataset[idx][0].permute(1, 2, 0).numpy()
        # Define the mean and standard deviation used for normalization
        mean = np.array([0.482, 0.482, 0.482])
        std = np.array([0.222, 0.222, 0.222])
        # Un-normalize the image using the defined mean and standard deviation
        img = std * img + mean
        # Clip the pixel values to a valid range
        img = np.clip(img, 0, 1)
        
        # Display the image on the subplot
        ax.imshow(img)
        # Turn off the axis to hide ticks and labels
        ax.axis('off')
        
        # Get the true and predicted class labels
        true_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        
        # Determine color for the predicted label for visual feedback
        pred_color = 'green' if true_label == pred_label else 'red'
        
        # Use ax.text for precise label placement on separate lines
        # Add the true label text to the plot
        ax.text(0.5, 1.05, f"True: {true_label}", 
                ha='center', transform=ax.transAxes, fontsize=12)
        # Add the predicted label text to the plot with the determined color
        ax.text(0.5, -0.1, f"Predicted: {pred_label}", 
                ha='center', transform=ax.transAxes, fontsize=12, color=pred_color)

    # Adjust layout to reduce whitespace
    plt.tight_layout(rect=[0, 0.03, 0.8, 0.95], h_pad=1.5)
    # Show the plot
    plt.show()
    
    
    
def display_prediction(trained_model, data_module, image_path):
    """
    Loads an image from a file path, makes a prediction using a trained model object,
    and displays the image with its predicted class.

    Args:
        trained_model: The trained Lightning model object.
        data_module: The data module that contains the dataset information.
        image_path: The file path to the image to be predicted.
    """
    # Retrieve class names from the data module's training dataset
    # data_module.setup()
    class_names = data_module.train_dataset.classes

    # Define the necessary image transformations
    transform = transforms.Compose([
        # Resize the image to a fixed size
        transforms.Resize((224, 224)),
        # Convert the image to a PyTorch tensor
        transforms.ToTensor(),
        # Normalize the tensor with the same values used in training
        transforms.Normalize([0.482, 0.482, 0.482], [0.222, 0.222, 0.222])
    ])

    # Load and preprocess the image
    try:
        # Open the image file and convert it to RGB format
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        # Handle the case where the image file is not found
        print(f"Error: Image not found at {image_path}")
        return

    # Apply transformations and add a batch dimension
    image_tensor = transform(image).unsqueeze(0)  

    # Ensure the model is on the correct device and in evaluation mode
    # Determine the device to use
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move the model to the selected device
    trained_model.to(device)
    # Set the model to evaluation mode
    trained_model.eval()
    # Move the image tensor to the same device as the model
    image_tensor = image_tensor.to(device)

    # Make the prediction
    # Disable gradient calculations
    with torch.no_grad():
        # Perform a forward pass to get model outputs
        outputs = trained_model(image_tensor)
        # Get the predicted class by finding the index with the highest value
        preds = torch.argmax(outputs, dim=1).item()

    # Get the predicted label name from the class names list
    predicted_label = class_names[preds]

    # Display the image and prediction
    # Display the image using matplotlib
    plt.imshow(image)
    # Set the title of the plot with the predicted label
    plt.title(f"Predicted: {predicted_label}")
    # Turn off the axis to hide ticks and labels
    plt.axis('off')
    # Show the plot
    plt.show()

    
    
def setup_dummy_weights(path="dummy_weights.pth", num_classes=3):
    """
    Creates and saves a dummy state_dict for a ResNet model.
    
    Args:
        path: The file path to save the dummy weights.
        num_classes: The number of output classes for the final layer.
    """
    # Initialize a ResNet-18 model without pre-trained weights
    model = tv_models.resnet18(weights=None)
    # Get the number of input features for the final fully connected layer
    num_ftrs = model.fc.in_features
    # Replace the final layer with a new one that has the specified number of output classes
    model.fc = nn.Linear(num_ftrs, num_classes)
    # Save the model's state dictionary to the specified path
    torch.save(model.state_dict(), path)
    # Return the path where the weights were saved
    return path



def cleanup_dummy_weights(path="dummy_weights.pth"):
    """
    Removes the dummy weights file.

    Args:
        path: The file path of the dummy weights to remove.
    """
    # Check if the file exists at the specified path
    if os.path.exists(path):
        # Remove the file
        os.remove(path)