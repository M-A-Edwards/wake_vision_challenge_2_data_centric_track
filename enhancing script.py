import os
import shutil
from datetime import datetime
from ultralytics import YOLO
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

def process_mislabeled_images(model_path: str, dataset_root: str, confidence_threshold: float = 0.4) -> tuple[int, str]:
    """
    Scans a dataset directory for mislabeled images in the non-person directory using a YOLO model,
    moves detected person images to the person directory, and logs the results.

    Args:
        model_path (str): Path to the YOLO model file (e.g., 'yolo11x.pt').
        dataset_root (str): Root directory of the dataset containing '0' (non-person) and '1' (person) subdirectories.
        confidence_threshold (float, optional): Confidence threshold for person detection. Defaults to 0.4.

    Returns:
        tuple[int, str]: A tuple containing the number of moved images and the path to the log file.

    When scaling, a smaller model with lowered threshold can be used.
    """
    try:
        model = YOLO(model_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found at {model_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {e}")

    non_person_dir = os.path.join(dataset_root, "0")
    person_dir = os.path.join(dataset_root, "1")
    logs_dir = os.path.join(dataset_root, "logs")

    for directory in [non_person_dir, person_dir]:
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Dataset directory not found: {directory}")

    os.makedirs(logs_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(logs_dir, f"moved_files_{timestamp}.txt")
    moved_count = 0

    print("Starting scan for mislabeled images in non-person directory...")

    with open(log_file_path, "w") as log_file:
        log_file.write("Moved Files (Person Detected in Non-Person Directory):\n\n")
        for filename in os.listdir(non_person_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(non_person_dir, filename)
                try:
                    results = model(file_path, verbose=False)
                    for result in results:
                        for box in result.boxes:
                            class_id = int(box.cls)
                            confidence = float(box.conf)
                            if class_id == 0 and confidence > confidence_threshold:
                                new_path = os.path.join(person_dir, filename)
                                shutil.move(file_path, new_path)
                                log_file.write(f"Moved {filename} (confidence: {confidence:.2f})\n")
                                moved_count += 1
                                print(f"Moved {filename} to person directory (confidence: {confidence:.2f})")
                                break
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
                    log_file.write(f"Error processing {filename}: {e}\n")

    print(f"\nCompleted processing. Moved {moved_count} mislabeled images to person directory.")
    print(f"Log file saved to: {log_file_path}")

    return moved_count, log_file_path

def visualize_mislabeled_images(model_path: str, dataset_root: str, confidence_threshold: float = 0.4, max_images: int = 20) -> None:
    """
    Scans the non-person directory for mislabeled images containing persons using a YOLO model,
    and visualizes up to a specified number of detected mislabels with their filenames and confidence scores.

    Args:
        model_path (str): Path to the YOLO model file (e.g., 'yolo11x.pt').
        dataset_root (str): Root directory of the dataset containing '0' (non-person) subdirectory.
        confidence_threshold (float, optional): Confidence threshold for person detection. Defaults to 0.4.
        max_images (int, optional): Maximum number of mislabeled images to visualize. Defaults to 20.

    Returns:
        None: Displays a grid of mislabeled images with filenames and confidence scores.

    """
    try:
        model = YOLO(model_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file not found at {model_path}: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {e}")

    non_person_dir = os.path.join(dataset_root, "0")
    if not os.path.exists(non_person_dir):
        raise FileNotFoundError(f"Non-person directory not found: {non_person_dir}")

    mislabelled = []
    print("Starting scan for mislabeled images in non-person directory...")

    for filename in os.listdir(non_person_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            file_path = os.path.join(non_person_dir, filename)
            try:
                results = model(file_path, verbose=False)
                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls)
                        confidence = float(box.conf)
                        if class_id == 0 and confidence > confidence_threshold:
                            mislabelled.append((file_path, filename, confidence))
                            break
                if len(mislabelled) >= max_images:
                    break
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nFound {len(mislabelled)} likely mislabeled images.")

    if mislabelled:
        plt.figure(figsize=(15, 8))
        for i, (path, filename, conf) in enumerate(mislabelled):
            try:
                image = Image.open(path)
                plt.subplot(4, 5, i + 1)
                plt.imshow(image)
                plt.axis("off")
                plt.title(f"{filename}\nconf: {conf:.2f}", fontsize=8)
            except Exception as e:
                print(f"Error displaying {filename}: {e}")
        plt.tight_layout()
        plt.show()
    else:
        print("No mislabeled images found to visualize.")

def get_advanced_augmentation():
    """
    Creates a TensorFlow Sequential model for advanced data augmentation, including random
    transformations to enhance dataset variability for training.

    Returns:
        tf.keras.Sequential: A Sequential model with configured augmentation layers.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.05, 0.05),
        tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.1),
    ])

def visualize_augmentation(dataset, augmentation):
    """
    Visualizes a batch of augmented images from the dataset for inspection.

    Args:
        dataset (tf.data.Dataset): The input dataset containing images and labels.
        augmentation (tf.keras.Sequential): The augmentation pipeline to apply.

    Returns:
        None: Displays a 3x3 grid of augmented images.
    """
    for images, labels in dataset.take(1):
        augmented_images = augmentation(images, training=True)
        plt.figure(figsize=(10, 10))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(augmented_images[i].numpy().astype("uint8"))
            plt.axis("off")
        plt.show()

if __name__ == "__main__":
    """
    Main execution block to process mislabeled images in the dataset.
    """
    model_path = "yolo11x.pt"
    dataset_root = "./wake_vision/train_quality"
    try:
        moved_count, log_file_path = process_mislabeled_images(model_path, dataset_root)
        # visualize_mislabeled_images(model_path, dataset_root) # Use this to visualize set number of mislabels
    except Exception as e:
        print(f"Error during mislabel processing: {e}")
