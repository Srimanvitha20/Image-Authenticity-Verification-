import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -------------------------------
# PATHS
# -------------------------------
TRAIN_DIR = r"C:\Users\suren\OneDrive\Desktop\image authenticity\dataset\ai-vs-real-images\Data Set 1\Data Set 1\train"
MODEL_SAVE_PATH = r"C:\Users\suren\OneDrive\Desktop\image authenticity\model\image_authenticity.keras"

# -------------------------------
# SETTINGS
# -------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 8   # SMALL to avoid memory error
EPOCHS = 10

# -------------------------------
# DATA LOADER
# -------------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training",
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation",
    shuffle=False
)

print("\n✅ Class Index Mapping:")
print(train_gen.class_indices)

# -------------------------------
# MODEL (TRANSFER LEARNING)
# -------------------------------
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # freeze base

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------
# CALLBACKS
# -------------------------------
checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

# -------------------------------
# TRAIN
# -------------------------------
print("\n🚀 Training Started...")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stop]
)

print("\n✅ MODEL TRAINED & SAVED SUCCESSFULLY!")
print(f"📌 Saved at: {MODEL_SAVE_PATH}")
