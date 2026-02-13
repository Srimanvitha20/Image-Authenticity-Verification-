import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import sys
import os

MODEL_PATH = r"C:\Users\suren\OneDrive\Desktop\image authenticity\model\image_authenticity.h5"
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 224

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    real_prob = model.predict(img_array)[0][0]
    fake_prob = 1 - real_prob

    real_percent = real_prob * 100
    fake_percent = fake_prob * 100

    return fake_percent, real_percent


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_image.py <image_path>")
        sys.exit()

    img_path = sys.argv[1]

    if not os.path.exists(img_path):
        print(" Image not found!")
        sys.exit()

    fake_p, real_p = predict_image(img_path)

    print("\n IMAGE AUTHENTICITY RESULT")
    print("---------------------------")
    print(f"Fake / AI Content : {fake_p:.2f}%")
    print(f"Real Content      : {real_p:.2f}%")

    if fake_p > 50:
        print(" Verdict: LIKELY AI / MORPHED IMAGE")
    else:
        print(" Verdict: LIKELY REAL IMAGE")
