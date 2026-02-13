import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os

# ---------------------------
# MODEL PATH
# ---------------------------
MODEL_PATH = r"C:\Users\suren\OneDrive\Desktop\image authenticity\model\image_authenticity.keras"

IMG_SIZE = 224

# Load model only once
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_pil_image(pil_img):
    """Convert PIL image to model input array"""
    img = pil_img.convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_full_image(pil_img):
    """Predict fake probability for full image"""
    x = preprocess_pil_image(pil_img)
    pred = model.predict(x, verbose=0)[0][0]  # sigmoid output
    return float(pred)  # fake probability


def split_into_patches(pil_img, grid=4):
    """
    Split image into patches.
    grid=4 => 4x4 = 16 patches
    """
    img = pil_img.convert("RGB")
    w, h = img.size

    patch_w = w // grid
    patch_h = h // grid

    patches = []
    for row in range(grid):
        for col in range(grid):
            left = col * patch_w
            upper = row * patch_h
            right = left + patch_w
            lower = upper + patch_h

            patch = img.crop((left, upper, right, lower))
            patches.append(patch)

    return patches


def predict_patch_based(pil_img, grid=4):
    """
    Predict fake probability on patches.
    Returns:
        avg_fake_prob, patch_probs(list)
    """
    patches = split_into_patches(pil_img, grid=grid)

    patch_probs = []
    for p in patches:
        prob = predict_full_image(p)
        patch_probs.append(prob)

    avg_fake = float(np.mean(patch_probs))
    return avg_fake, patch_probs


def final_verdict(fake_prob, morph_score):
    """
    We give ONLY ONE final answer:
    REAL / FAKE / MORPHED
    """

    # Thresholds (can be tuned)
    FAKE_HIGH = 0.65
    REAL_LOW = 0.35

    # If very fake
    if fake_prob >= FAKE_HIGH:
        return "FAKE / AI GENERATED"

    # If very real
    if fake_prob <= REAL_LOW:
        return "REAL IMAGE"

    # Otherwise uncertain → morphed/edited
    return "MORPHED / PARTIALLY EDITED"


def analyze_image(pil_img):
    """
    Final combined prediction:
    - Full image probability
    - Patch-based morph score
    - Single final verdict
    """

    # Full prediction
    fake_prob_full = predict_full_image(pil_img)

    # Patch prediction
    avg_fake_patch, patch_probs = predict_patch_based(pil_img, grid=4)

    # Morph score = how inconsistent patches are
    # More variance => more likely morphed
    morph_score = float(np.std(patch_probs))

    verdict = final_verdict(fake_prob_full, morph_score)

    return {
        "fake_percent": fake_prob_full * 100,
        "real_percent": (1 - fake_prob_full) * 100,
        "verdict": verdict,
        "morph_score": morph_score,
        "patch_probs": patch_probs
    }