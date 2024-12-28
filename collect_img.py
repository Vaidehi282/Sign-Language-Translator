import os
import cv2 as cv

data_dir = './data'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

number_of_classes = 26
dataset_size = 100

cap = cv.VideoCapture(1)

for j in range(number_of_classes):
    if not os.path.exists(os.path.join(data_dir, str(j))):
        os.makedirs(os.path.join(data_dir, str(j)))
    
    print('Collecting data for class ', j)

    done = False
    while True:
        ret, frame = cap.read()
        frame_flip = cv.flip(frame, 1)
        cv.putText(frame_flip, 'Ready press "S": ', (100,50), cv.FONT_HERSHEY_SIMPLEX, 1.3, (255,255,0), 3, cv.LINE_AA)
        cv.imshow('frame',frame_flip)
        if cv.waitKey(25) == ord('s'):
            break

    counter = 0
    
    while counter < dataset_size:
        ret, frame = cap.read()
        frame_flip = cv.flip(frame, 1)

        cv.imshow('frame', frame_flip)
        cv.waitKey(25)

        cv.imwrite(os.path.join(data_dir, str(j), f'{counter}.jpg'), frame_flip)
        counter += 1

cap.release()
cv.destroyAllWindows()
