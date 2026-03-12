"""
Garbage Detection Model Training Script
Uses MobileNetV2 transfer learning for classifying garbage categories.
Categories: cardboard, glass, metal, paper, plastic, trash
Dataset: TrashNet (or similar)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
NUM_CLASSES = 6
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def build_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    # Freeze base layers
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def train(data_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    model = build_model()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        ModelCheckpoint('garbage_model.h5', save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=7, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

    # Save class indices
    with open('class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f)

    print("Training complete. Model saved as garbage_model.h5")
    return model

if __name__ == '__main__':
    # Usage: python train_model.py
    # Place your dataset in a folder with subfolders per class
    data_dir = input("Enter path to dataset directory: ")
    train(data_dir)
