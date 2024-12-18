import tensorflow as tf
import numpy as np
import os
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Map labels to integers
LABEL_MAP = {
    "barking": 1,
    "non_barking": 0
}

# ------------------------------
# Load Spectrogram Data
# ------------------------------
def load_data_from_npy(data_dir):
    spectrograms = []
    labels = []
    for label_name, label_value in {"barking": 1, "non_barking": 0}.items():
        label_dir = os.path.join(data_dir, label_name)
        if not os.path.exists(label_dir):
            print(f"Label directory '{label_dir}' does not exist. Skipping...")
            continue
        #iterate through directory
        for file in os.listdir(label_dir):
            if file.endswith('.npy'):
                file_path = os.path.join(label_dir, file)
                spectrogram = np.load(file_path)
                # Convert to float32 to reduce memory usage by half
                spectrogram = spectrogram.astype(np.float32)
                spectrogram = np.expand_dims(spectrogram, axis=-1)
                spectrograms.append(spectrogram)
                labels.append(label_value)

    # Return after both classes are processed
    return np.array(spectrograms), np.array(labels)

# ------------------------------
# Build CNN Model
# ------------------------------
def build_cnn(input_shape):
    model = models.Sequential([
        Input(shape=input_shape),  # Input layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),  # Reduced from 0.5
        layers.Dense(1, activation='sigmoid')   #output layer, using sigmoid activation due to binary classifier

    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy']) #predict label via binary cross entropy
    return model

# ------------------------------
# Train Model
# ------------------------------
def train_model(data_dir, batch_size=16, epochs=10):
    """
    Train the CNN model using datasets loaded from .npy files.
    """
    # Load data
    spectrograms, labels = load_data_from_npy(data_dir)
    
    # Split the data 80/20(70 30 had lower accuracy)
    X_train, X_val, y_train, y_val = train_test_split(
        spectrograms, labels, test_size=0.2, random_state=42
    )
    print(f"Dataset sizes - Total: {len(labels)}, Train: {len(y_train)}, Validation: {len(y_val)}")

    # Increased the weight for barking to encourage the model to pay more attention to it.
    class_weight_dict = {
        0: 1.0,  # weight for non_barking(same value as orignal)
        1: 2.0   # weight for barking (double weight)
    }
    print("Class weights:", class_weight_dict)

    # Convert to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    # Batch and prefetch datasets
    train_dataset = train_dataset.shuffle(len(y_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Build and compile the model
    input_shape = (512, 512, 1)  #image size plus the extra channel
    model = build_cnn(input_shape)

    # Train the model
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        class_weight=class_weight_dict
    )
    # Save the model
    model.save("barking_detection_model.keras")

    print("Model saved as 'barking_detection_model.keras'")

    return model, history




# ------------------------------
# Evaluate Model
# ------------------------------
def evaluate_model(model, val_dataset):
    """
    Evaluate the trained model on validation data.
    """
    loss, accuracy = model.evaluate(val_dataset)
    print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")

# ------------------------------
# Main Script
# ------------------------------
if __name__ == "__main__":
    data_dir = "data/spectrograms"  # Path to the data directory containing .npy files

    # Train the model
    model, history = train_model(data_dir)

    # Plot training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    else:
        print("Validation accuracy not found in history. Skipping plot for validation.")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training vs Validation Accuracy')
    plt.show()
    #Plot overfitting
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training vs Validation Loss')
    plt.show()


