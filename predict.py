import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ---------------------------
# 1) LOAD MODEL
# ---------------------------
MODEL_PATH = r"C:\Users\suren\OneDrive\Desktop\image authenticity\model\image_authenticity.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model Loaded Successfully!")


# ---------------------------
# 2) IMAGE PATH (PUT YOUR TEST IMAGE HERE)
# ---------------------------
IMAGE_PATH = r"C:\Users\suren\OneDrive\Desktop\image authenticity\test_images\image 1.jpg"

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"❌ Image not found at: {IMAGE_PATH}")


# ---------------------------
# 3) FUNCTION: Predict a single patch
# ---------------------------
def predict_patch(patch_array):
    """
    patch_array: numpy array shape (224,224,3)
    returns fake_prob (0 to 1)
    """
    patch_array = patch_array.astype(np.float32) / 255.0
    patch_array = np.expand_dims(patch_array, axis=0)

    pred = model.predict(patch_array, verbose=0)[0][0]
    return float(pred)  # fake probability


# ---------------------------
# 4) FUNCTION: Scan image in patches (to detect morph)
# ---------------------------
def patch_scan(image_path, patch_size=224, step=112):
    """
    Returns:
        avg_fake_prob (0 to 1)
    """
    img = image.load_img(image_path)
    img_arr = image.img_to_array(img)

    h, w, _ = img_arr.shape
    fake_scores = []

    for y in range(0, h - patch_size + 1, step):
        for x in range(0, w - patch_size + 1, step):
            patch = img_arr[y:y + patch_size, x:x + patch_size]
            fake_prob = predict_patch(patch)
            fake_scores.append(fake_prob)

    # If image is too small, fallback to resize prediction
    if len(fake_scores) == 0:
        img_resized = image.load_img(image_path, target_size=(224, 224))
        img_resized = image.img_to_array(img_resized)
        fake_prob = predict_patch(img_resized)
        return fake_prob

    return float(np.mean(fake_scores))


# ---------------------------
# 5) FINAL VERDICT FUNCTION (ONLY ONE OUTPUT)
# ---------------------------
def final_verdict(fake_percent):
    """
    fake_percent: 0 to 100
    returns ONLY ONE label
    """
    if fake_percent >= 70:
        return "FAKE / AI GENERATED"
    elif fake_percent <= 30:
        return "REAL IMAGE"
    else:
        return "MORPHED / PARTIALLY EDITED"


avg_fake_prob = patch_scan(IMAGE_PATH)
fake_percent = avg_fake_prob * 100
real_percent = 100 - fake_percent

verdict = final_verdict(fake_percent)

print("\nIMAGE AUTHENTICITY RESULT (FINAL)")
print("-----------------------------------")
print(f"Final Verdict: ✅ {verdict}")
print(f"(Confidence Reference: Fake = {fake_percent:.2f}%, Real = {real_percent:.2f}%)")
