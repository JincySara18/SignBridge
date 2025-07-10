# added by Jincy June-2nd-2025
import cv2
import mediapipe as mp
import pandas as pd
import os
import time

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Output CSV file
csv_file = "hand_landmarks.csv"
columns = [f'{dim}{i}' for i in range(21) for dim in ('x', 'y', 'z')] + ['label']

# Create file with header if it doesn't exist
if not os.path.exists(csv_file):
    pd.DataFrame(columns=columns).to_csv(csv_file, index=False)

# Label mapping keys
label_map = {
    ord('y'): "Yes",
    ord('h'): "Hello",
    ord('t'): "Thank you",
    ord('i'): "I",
    ord('u'): "You",
    ord('a'): "Hate",
    ord('l'): "Love"
}

# Start webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press a key to start capturing for that sign ('y', 'h', 't', etc.) or 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Capture", img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key in label_map:
        label = label_map[key]
        print(f"[INFO] Capturing 30 samples for label: {label}")
        captured = 0
        while captured < 30:
            success, img = cap.read()
            if not success:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    landmark_list = []

                    # Normalize to wrist (landmark 0)
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y
                    base_z = hand_landmarks.landmark[0].z

                    for lm in hand_landmarks.landmark:
                        landmark_list.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])

                    landmark_list.append(label)
                    df = pd.DataFrame([landmark_list], columns=columns)
                    df.to_csv(csv_file, mode='a', header=False, index=False)
                    print(f"[SAVED] {label} sample {captured+1}")
                    captured += 1

            cv2.imshow("Capture", img)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("[INFO] Data capture completed and saved to CSV.")
