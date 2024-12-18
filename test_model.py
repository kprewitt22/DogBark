import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import os

# Path to the saved model
MODEL_PATH = "barking_detection_model.keras"

# Path to validation dataset directory
DATA_DIR = "data/spectrograms"  
BATCH_SIZE = 16

def load_data_generator(data_dir, batch_size):
    """
    A generator to load spectrograms and labels in batches.
    """
    label_map = {"barking": 1, "non_barking": 0}
    spectrogram_files = []
    labels = []

    # Gather file paths and labels
    for label_name, label_value in label_map.items():
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.exists(label_dir):
            print(f"Warning: Directory '{label_dir}' does not exist. Skipping...")
            continue
        files = [f for f in os.listdir(label_dir) if f.endswith('.npy')]
        if not files:
            print(f"Warning: Directory '{label_dir}' is empty. Skipping...")
            continue
        spectrogram_files.extend([os.path.join(label_dir, f) for f in files])
        labels.extend([label_value] * len(files))

    # Shuffle data
    combined = list(zip(spectrogram_files, labels))
    np.random.shuffle(combined)
    spectrogram_files, labels = zip(*combined)

    # Yield batches
    for i in range(0, len(spectrogram_files), batch_size):
        batch_files = spectrogram_files[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

        batch_data = [np.expand_dims(np.load(file), axis=-1) for file in batch_files]
        yield np.array(batch_data), np.array(batch_labels)

def evaluate_model(model_path, data_dir, batch_size):
    # Load the trained model
    try:
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize the data generator
    data_generator = load_data_generator(data_dir, batch_size)

    true_labels = []
    predicted_labels = []

    print("\nEvaluating the model on validation data...")
    for batch_data, batch_labels in data_generator:
        predictions = model.predict(batch_data, verbose=0)
        predicted_labels.extend((predictions > 0.5).astype(int).flatten())
        true_labels.extend(batch_labels)

    # Compute accuracy
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Generate classification report and confusion matrix
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, target_names=['Non-Barking', 'Barking']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(true_labels, predicted_labels))

    # Class distribution debugging
    barking_count = len([f for f in os.listdir(os.path.join(data_dir, 'barking')) if f.endswith('.npy')])
    non_barking_count = len([f for f in os.listdir(os.path.join(data_dir, 'non_barking')) if f.endswith('.npy')])
    print(f"\nClass Distribution in Validation Set:")
    print(f"Barking: {barking_count}, Non-Barking: {non_barking_count}")


if __name__ == "__main__":
    evaluate_model(MODEL_PATH, DATA_DIR, BATCH_SIZE)
