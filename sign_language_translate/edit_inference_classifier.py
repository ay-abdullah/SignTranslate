import pickle
import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./models.p', 'rb'))
model_rfc = model_dict['model_rfc']
model_knn = model_dict['model_knn']
model_svm = model_dict['model_svm']


cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'Ara Beni',
    1: 'Dostluk',
    2: 'Isaret Etme',
    3: 'Dikkat',
    4: 'Dur,Bekle',
    5: 'Eglence',
    6: 'Guc,Birlik',
    7: 'Dayanisma,Direnis',
    8: 'Her sey yolunda, Tamam',
    9: 'Reddetme, Hayir',
    10: 'Merhaba, Selam',
    11: 'Onaylama, Begenme',
    12: 'Reddetme',
    13: 'Uzun ve Basarili bir yasam dileme',
    14: 'Yazi yazma',
    15: 'Yol gosterme',
    16: 'Umut etmek',
}

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        # Eğer sadece bir el varsa işlemi yap
        if len(results.multi_hand_landmarks) == 1:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10

            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            try:
                prediction = model_rfc.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)
            except Exception as e:
                print(f"Prediction error: {e}")
                cv2.putText(frame, 'Prediction error', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3,
                            cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
