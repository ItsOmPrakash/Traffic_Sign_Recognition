import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# --- CONFIG ---
MODEL_PATH = 'traffic_sign_model_FIXED.keras'
TEST_IMAGE = 'test_stop.jpg' # Make sure you have this file!
IMAGE_SIZE = (30, 30)

# --- LOAD MODEL ---
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- PREPROCESS FUNCTION (Must be identical to training) ---
def test_prediction(image_path):
    # 1. Load using PIL (Same as training)
    try:
        img = Image.open(image_path).convert('RGB')
    except:
        print(f"Error: Could not open {image_path}")
        return

    # 2. Resize
    img = img.resize(IMAGE_SIZE)
    
    # 3. Convert to Array and Normalize
    img_array = np.array(img)
    img_array = img_array / 255.0
    
    # 4. Expand dims (1, 30, 30, 3)
    input_tensor = np.expand_dims(img_array, axis=0)

    # 5. Predict
    predictions = model.predict(input_tensor, verbose=0)[0]
    class_id = np.argmax(predictions)
    confidence = predictions[class_id]

    print(f"--- RESULTS FOR {image_path} ---")
    print(f"Predicted Class ID: {class_id}")
    print(f"Confidence Score: {confidence:.4f}")
    
    # Check if it works
    if confidence > 0.5:
        print("✅ SUCCESS: The model works on clear images!")
    else:
        print("❌ FAILURE: The model is broken (or needs more training).")

if __name__ == "__main__":
    test_prediction(TEST_IMAGE)