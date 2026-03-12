"""
Creates a demo garbage detection model using MobileNetV2 with random weights
for deployment testing when no trained weights are available.
Run this once to generate garbage_model.h5
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import json
import os

IMG_SIZE = 224
NUM_CLASSES = 6
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def create_model():
    base_model = MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    print("Creating demo model...")
    model = create_model()

    save_path = os.path.join(os.path.dirname(__file__), 'garbage_model.h5')
    model.save(save_path)
    print(f"Demo model saved to {save_path}")

    class_indices = {cls: i for i, cls in enumerate(CLASSES)}
    idx_path = os.path.join(os.path.dirname(__file__), 'class_indices.json')
    with open(idx_path, 'w') as f:
        json.dump(class_indices, f)
    print(f"Class indices saved to {idx_path}")
