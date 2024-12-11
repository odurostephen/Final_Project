# Libraries and Setup
import os
import glob
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization,
                                     Activation, Input, concatenate, Dropout)
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tarfile
from pyspark.sql import SparkSession
import random
import cv2

# Create Spark session
spark = SparkSession.builder \
    .appName("BrainTumorClassification") \
    .getOrCreate()

# Paths and Data Extraction
base_path = 'soduro/data/'  
extracted_path = './data'
os.makedirs(extracted_path, exist_ok=True)

tar_files = glob.glob(os.path.join(base_path, "*.tar"))
for tar_file in tar_files:
    with tarfile.open(tar_file) as tar:
        tar.extractall(extracted_path)
        print(f"Extracted {tar_file}")

# Data Preprocessing
def load_nifti_file(file_path):
    nifti_image = nib.load(file_path)
    return np.asarray(nifti_image.dataobj)

def preprocess_modalities(brain_dir):
    modalities = []
    for mod in ['flair', 't1', 't1ce', 't2', 'seg']:
        file_path = glob.glob(os.path.join(brain_dir, f"*_{mod}.nii.gz"))[0]
        modalities.append(load_nifti_file(file_path))
    modalities = np.stack(modalities, axis=-1)
    return modalities

input_data = []
path = extracted_path
subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
for subdir in subdirs:
    brain_dir = os.path.join(path, subdir)
    processed_data = preprocess_modalities(brain_dir)
    input_data.append(processed_data)

input_data = np.array(input_data, dtype=np.float32)
print(f"Input data shape: {input_data.shape}")

# Data Augmentation
def augment_data(data, labels):
    augmented_data, augmented_labels = [], []
    for i in range(data.shape[0]):
        image, label = data[i], labels[i]
        if random.random() > 0.5:
            image = np.flip(image, axis=1)
            label = np.flip(label, axis=1)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            image = np.rot90(image, k=angle // 90, axes=(0, 1))
            label = np.rot90(label, k=angle // 90, axes=(0, 1))
        augmented_data.append(image)
        augmented_labels.append(label)
    return np.array(augmented_data), np.array(augmented_labels)

X_augmented, Y_augmented = augment_data(input_data[..., :4], input_data[..., 4])

# Data Splitting
X_train, X_temp, Y_train, Y_temp = train_test_split(X_augmented, Y_augmented, test_size=0.3, random_state=42)
X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.5, random_state=42)
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# CNN (U-Net) Model
def convolution_block(input_tensor, filters):
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_tensor)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    return x

def build_unet(input_shape):
    inputs = Input(input_shape)

    conv1 = convolution_block(inputs, 32)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = convolution_block(pool1, 64)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = convolution_block(pool2, 128)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = convolution_block(pool3, 256)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = convolution_block(pool4, 512)

    up6 = UpSampling2D((2, 2))(conv5)
    merge6 = concatenate([up6, conv4])
    conv6 = convolution_block(merge6, 256)

    up7 = UpSampling2D((2, 2))(conv6)
    merge7 = concatenate([up7, conv3])
    conv7 = convolution_block(merge7, 128)

    up8 = UpSampling2D((2, 2))(conv7)
    merge8 = concatenate([up8, conv2])
    conv8 = convolution_block(merge8, 64)

    up9 = UpSampling2D((2, 2))(conv8)
    merge9 = concatenate([up9, conv1])
    conv9 = convolution_block(merge9, 32)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    return Model(inputs, outputs)

model = build_unet((128, 128, 4))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Training the Model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=30,
    batch_size=16
)

# Evaluation and Results
test_loss, test_accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Save Model
model.save("brain_tumor_unet_model.h5")
print("Model saved successfully!")

