import cv2
import numpy as np
import tensorflow as tf

# --- CONFIGURATION ---
# Use the new, fixed model filename
MODEL_PATH = 'traffic_sign_model_FIXED.keras' 
IMAGE_SIZE = (30, 30)
THRESHOLD = 0.45 # Confidence threshold (45%)

# --- 1. LOAD THE MODEL ---
print("Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except OSError:
    print(f"Error: Could not find file '{MODEL_PATH}'. Check your folder.")
    exit()

# --- 2. LABELS DICTIONARY (Pasted from your output) ---
LABELS = {
    0: 'ALL_MOTOR_VEHICLE_PROHIBITED', 1: 'AXLE_LOAD_LIMIT', 2: 'BARRIER_AHEAD', 
    3: 'BULLOCK_AND_HANDCART_PROHIBITED', 4: 'BULLOCK_PROHIBITED', 5: 'CATTLE', 
    6: 'COMPULSARY_AHEAD', 7: 'COMPULSARY_AHEAD_OR_TURN_LEFT', 8: 'COMPULSARY_AHEAD_OR_TURN_RIGHT', 
    9: 'COMPULSARY_CYCLE_TRACK', 10: 'COMPULSARY_KEEP_LEFT', 11: 'COMPULSARY_KEEP_RIGHT', 
    12: 'COMPULSARY_MINIMUM_SPEED', 13: 'COMPULSARY_SOUND_HORN', 14: 'COMPULSARY_TURN_LEFT', 
    15: 'COMPULSARY_TURN_LEFT_AHEAD', 16: 'COMPULSARY_TURN_RIGHT', 17: 'COMPULSARY_TURN_RIGHT_AHEAD', 
    18: 'CROSS_ROAD', 19: 'CYCLE_CROSSING', 20: 'CYCLE_PROHIBITED', 21: 'DANGEROUS_DIP', 
    22: 'DIRECTION', 23: 'FALLING_ROCKS', 24: 'FERRY', 25: 'GAP_IN_MEDIAN', 26: 'GIVE_WAY', 
    27: 'GUARDED_LEVEL_CROSSING', 28: 'HANDCART_PROHIBITED', 29: 'HEIGHT_LIMIT', 
    30: 'HORN_PROHIBITED', 31: 'HUMP_OR_ROUGH_ROAD', 32: 'LEFT_HAIR_PIN_BEND', 
    33: 'LEFT_HAND_CURVE', 34: 'LEFT_REVERSE_BEND', 35: 'LEFT_TURN_PROHIBITED', 
    36: 'LENGTH_LIMIT', 37: 'LOAD_LIMIT', 38: 'LOOSE_GRAVEL', 39: 'MEN_AT_WORK', 
    40: 'NARROW_BRIDGE', 41: 'NARROW_ROAD_AHEAD', 42: 'NO_ENTRY', 43: 'NO_PARKING', 
    44: 'NO_STOPPING_OR_STANDING', 45: 'OVERTAKING_PROHIBITED', 46: 'PASS_EITHER_SIDE', 
    47: 'PEDESTRIAN_CROSSING', 48: 'PEDESTRIAN_PROHIBITED', 49: 'PRIORITY_FOR_ONCOMING_VEHICLES', 
    50: 'QUAY_SIDE_OR_RIVER_BANK', 51: 'RESTRICTION_ENDS', 52: 'RIGHT_HAIR_PIN_BEND', 
    53: 'RIGHT_HAND_CURVE', 54: 'RIGHT_REVERSE_BEND', 55: 'RIGHT_TURN_PROHIBITED', 
    56: 'ROAD_WIDENS_AHEAD', 57: 'ROUNDABOUT', 58: 'SCHOOL_AHEAD', 59: 'SIDE_ROAD_LEFT', 
    60: 'SIDE_ROAD_RIGHT', 61: 'SLIPPERY_ROAD', 62: 'SPEED_LIMIT_15', 63: 'SPEED_LIMIT_20', 
    64: 'SPEED_LIMIT_30', 65: 'SPEED_LIMIT_40', 66: 'SPEED_LIMIT_5', 67: 'SPEED_LIMIT_50', 
    68: 'SPEED_LIMIT_60', 69: 'SPEED_LIMIT_70', 70: 'SPEED_LIMIT_80', 71: 'STAGGERED_INTERSECTION', 
    72: 'STEEP_ASCENT', 73: 'STEEP_DESCENT', 74: 'STOP', 75: 'STRAIGHT_PROHIBITED', 
    76: 'TONGA_PROHIBITED', 77: 'TRAFFIC_SIGNAL', 78: 'TRUCK_PROHIBITED', 79: 'TURN_RIGHT', 
    80: 'T_INTERSECTION', 81: 'UNGUARDED_LEVEL_CROSSING', 82: 'U_TURN_PROHIBITED', 
    83: 'WIDTH_LIMIT', 84: 'Y_INTERSECTION'
}

def preprocess_frame(frame):
    # CRITICAL: Convert BGR (OpenCV) to RGB (Model)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to 30x30
    img = cv2.resize(img, IMAGE_SIZE)
    # Normalize pixel values
    img = img / 255.0
    # Add batch dimension
    return np.expand_dims(img, axis=0)

# cv2.CAP_DSHOW helps Windows connect to the camera faster
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(3, 640) # Width
cap.set(4, 480) # Height

print("Starting Camera... Press 'q' to quit.")

while True:
    success, frame = cap.read()
    if not success:
        break

    # Define the "Region of Interest" (ROI) box
    h, w, _ = frame.shape
    x1, y1 = int(w/2 - 80), int(h/2 - 80)
    x2, y2 = int(w/2 + 80), int(h/2 + 80)

    # Extract the image inside the box
    roi = frame[y1:y2, x1:x2]
    
    if roi.size > 0:
        # Predict
        processed_roi = preprocess_frame(roi)
        predictions = model.predict(processed_roi, verbose=0)[0]
        
        class_id = np.argmax(predictions)
        confidence = predictions[class_id]
        
        # --- DEBUG PRINT ---
        class_name = LABELS.get(class_id, "Unknown ID")
        print(f"I see Class: {class_id} ({class_name}) | Confidence: {confidence:.2f}")

        label_text = "Unknown"
        color = (0, 0, 255) # Red by default

        if confidence > THRESHOLD:
            label_text = class_name
            color = (0, 255, 0) # Green for confident match
            
            # Display confidence score
            conf_text = f"{confidence*100:.1f}%"
            cv2.putText(frame, conf_text, (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw the box and the label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the frame
    cv2.imshow("Traffic Sign Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()