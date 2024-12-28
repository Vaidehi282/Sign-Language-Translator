import cv2 as cv
import mediapipe as mp
import pickle
import numpy as np
import string

model = pickle.load(open('model.p', 'rb'))
model = model['model']

predict_dict = {i: letter for i, letter in enumerate(string.ascii_uppercase[:-1])}

capture = cv.VideoCapture(1)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


while True:
    ret, frame = capture.read()
    frame_flip = cv.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv.cvtColor(frame_flip, cv.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        data_aux = []
        x_ = []
        y_ = []
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_flip,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)

        padding = 100
        x1 = int(min(x_) * w)
        y1 = int(min(y_) * h)
        x2 = int(max(x_) * w)
        y2 = int(max(y_) * h)

        predict = model.predict([np.asarray(data_aux)])
        predict_character = predict_dict[int(predict[0])]

        # if predict_character != last_character:
        #     sentence += predict_character
        #     last_character = predict_character

        cv.rectangle(frame_flip, (x1 + 10, y1 + 10),
                     (x2 + 10, y2 + 10), (255, 0, 0), 4)
        cv.putText(frame_flip, predict_character, (x1 + 10, y1 - 10),
                   cv.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 4, cv.LINE_AA)

        # cv.putText(frame_flip, f"Sentence: {sentence}", (20, 50),
        #        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow('frame', frame_flip)

    if cv.waitKey(25) == ord('c'):
        sentence = ''
    if cv.waitKey(25) == ord('q'):
        break
    

capture.release()
cv.destroyAllWindows()
