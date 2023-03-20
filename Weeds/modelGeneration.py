import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Définir les paramètres
batch_size = 32
epochs = 20 # Nombre d'itérations
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Définir les chemins d'accès aux données
train_dir = 'Weeds/dataset/train/' #replace by train_dir = "dataset/train/" if it doesn't work.
validation_dir = 'Weeds/dataset/validation/' #replace by validation_dir = "dataset/validation/" if it doesn't work.

# Prétraiter les images
train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range=45,
                                           width_shift_range=.15,
                                           height_shift_range=.15,
                                           horizontal_flip=True,
                                           zoom_range=0.5)
validation_image_generator = ImageDataGenerator(rescale=1./255)

# Charger les données d'entraînement et de validation
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

# Créer le modèle CNN
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

# Compiler le modèle
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Entraîner le modèle
history = model.fit(
    train_data_gen,
    steps_per_epoch=  train_data_gen.samples // batch_size,
    validation_data=val_data_gen,
    validation_steps= val_data_gen.samples // batch_size,
    epochs=epochs
)

# Sauvegarder le modèle
model.save("model.h5")

