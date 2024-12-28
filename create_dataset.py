import mediapipe as mp
import cv2 as cv
import os
import pickle

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

data_dir = './data'
data = []
labels = []
for dir in os.listdir(data_dir):
    if dir == '.DS_Store':
        continue
    for img_path in os.listdir(os.path.join(data_dir, dir)):
        if img_path == '.DS_Store':
            continue
        data_aux = []
        img = cv.imread(os.path.join(data_dir, dir, img_path))
        img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # mp_drawing.draw_landmarks(
                #     img_rgb,
                #     hand_landmarks,
                #     mp_hands.HAND_CONNECTIONS,
                #     mp_drawing_styles.get_default_hand_landmarks_style(),
                #     mp_drawing_styles.get_default_hand_connections_style()
                # )
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x)
                    data_aux.append(y)

            data.append(data_aux)
            labels.append(dir)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels':labels}, f)
                
        
