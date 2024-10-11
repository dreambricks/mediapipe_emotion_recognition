import pickle

import cv2

from utils import FaceLandmarks

fl = FaceLandmarks(static_image_mode=False)
emotions = ['happy', 'laughing', 'neutral'] # , 'laughing'
emotions_pt = ['sorrindo', 'rindo', 'neutro'] # , 'sor. leve'
emotions_idx_sorted = [1, 2, 0]
with open('./model3', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(2)

ret, frame = cap.read()

while ret:
    ret, frame = cap.read()

    face_landmarks = fl.get_face_landmarks(frame, draw=False)
    
    if len(face_landmarks) == 1404:
        #output = model.predict([face_landmarks])
        output = model.predict_proba([face_landmarks])
        #print(output[0])
        max_val = max(output[0])
        for idx, e_idx in enumerate(emotions_idx_sorted):
            e = emotions_pt[e_idx]
            text = f"{e} : {output[0][e_idx]*100:.0f}%"
            color = (0, 255, 0) if output[0][e_idx] == max_val else (0, 0, 255)
            cv2.putText(frame,
                        text,
                       (10, frame.shape[0] - 10 - (len(emotions)-idx-1)*35),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1,
                       color,
                       3)
            
        #cv2.putText(frame,
        #            emotions[int(output[0])],
        #           (10, frame.shape[0] - 1),
        #           cv2.FONT_HERSHEY_SIMPLEX,
        #           3,
        #           (0, 255, 0),
        #           5)

    cv2.imshow('frame', frame)

    cv2.waitKey(25)


cap.release()
cv2.destroyAllWindows()