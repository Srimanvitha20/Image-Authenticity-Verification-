import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os
from PIL import Image

MODEL_PATH = r"C:\Users\suren\OneDrive\Desktop\image authenticity\model\image_authenticity.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224


def preprocess_patch(patch):
    patch = patch.resize((IMG_SIZE, IMG_SIZE))
    patch_array = np.array(patch).astype("float32") / 255.0
    patch_array = np.expand_dims(patch_array, axis=0)
    return patch_array


def predict_morph_percentage(img_path, grid=3):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    patch_w = w // grid
    patch_h = h // grid

    fake_scores = []

    for row in range(grid):
        for col in range(grid):
            left = col * patch_w
            upper = row * patch_h
            right = (col + 1) * patch_w
            lower = (row + 1) * patch_h

            patch = img.crop((left, upper, right, lower))

            patch_array = preprocess_patch(patch)

            real_prob = model.predict(patch_array, verbose=0)[0][0]
            fake_prob = 1 - real_prob

            fake_scores.append(fake_prob)

    fake_percentage = np.mean(fake_scores) * 100
    real_percentage = 100 - fake_percentage

    return fake_percentage, real_percentage


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_morph_percentage.py <image_path>")
        sys.exit()

    img_path = sys.argv[1]

    if not os.path.exists(img_path):
        print(" Image not found!")
        sys.exit()

    fake_p, real_p = predict_morph_percentage(img_path, grid=3)

    print("\n MORPH / AI PERCENTAGE RESULT (Patch Based)")
    print("-------------------------------------------")
    print(f"AI / Fake Content : {fake_p:.2f}%")
    print(f"Real Content      : {real_p:.2f}%")

    if fake_p > 50:
        print(" Verdict: MOSTLY AI / MORPHED")
    elif fake_p > 20:
        print(" Verdict: PARTIALLY MORPHED / EDITED")
    else:
        print(" Verdict: MOSTLY REAL")
