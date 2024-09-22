import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import cv2
import os
from tensorflow.keras.utils import to_categorical


# Function to preprocess individual images
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 32))  # Resize to (128, 32)
    img = img.astype(np.float32) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=-1)  # Add channel dimension
    return img


# Function to load dataset from a specified path
def load_dataset(dataset_path):
    images = []
    labels = []
    label_map = {}
    current_label_id = 0

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            if label not in label_map:
                label_map[label] = current_label_id
                current_label_id += 1

            for filename in os.listdir(label_path):
                if filename.endswith(('.png', '.jpg')):
                    img_path = os.path.join(label_path, filename)
                    img = preprocess_image(img_path)
                    images.append(img)
                    labels.append(label_map[label])

    return np.array(images), np.array(labels), label_map


# Example usage
dataset_path = 'C:/Users/asus/Music/tensorflow project/archive (1)/data/testing_data'  # Path to your dataset
images, labels, label_map = load_dataset(dataset_path)
print(f"Loaded {len(images)} images with {len(set(labels))} unique labels from dataset.")
print(f"Label map: {label_map}")


# Function to build a CRNN model
def build_crnn_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # CNN part
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Reshape(target_shape=(-1, x.shape[-1]))(x)  # Reshape for LSTM input
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model


# Build and compile the model
num_classes = len(label_map)
input_shape = (32, 128, 1)  # Input shape for the model
model = build_crnn_model(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Prepare the dataset for training
one_hot_labels = to_categorical(labels, num_classes=num_classes)
split_idx = int(0.8 * len(images))  # 80-20 split for training and validation
train_images, val_images = images[:split_idx], images[split_idx:]
train_labels, val_labels = one_hot_labels[:split_idx], one_hot_labels[split_idx:]

# Train the model
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))
model.summary()


# Function for inference on a new image
def ocr_inference(model, image_path):
    processed_image = preprocess_image(image_path)
    processed_image = np.expand_dims(processed_image, axis=0)  # Add batch dimension
    predictions = model.predict(processed_image)
    decoded_text = decode_predictions(predictions)
    return decoded_text


# Function to decode predictions into text
def decode_predictions(predictions):
    predicted_indices = np.argmax(predictions, axis=-1)
    if isinstance(predicted_indices[0], (list, np.ndarray)):
        decoded_text = "".join([get_char_from_label(idx) for idx in predicted_indices[0]])
    else:
        decoded_text = get_char_from_label(predicted_indices[0])
    return decoded_text


# Function to map index back to character
def get_char_from_label(index):
    reverse_label_map = {v: k for k, v in label_map.items()}
    return reverse_label_map.get(index, "")


# Run inference on a sample image
image_path = 'C:/Users/asus/Music/tensorflow project/archive (1)/data/testing_data/0/28562.png'
predicted_text = ocr_inference(model, image_path)
print(f"Predicted Text: {predicted_text}")

# Save the trained model
model.save('C:/Users/asus/Music/tensorflow project/ocr_model.h5')
