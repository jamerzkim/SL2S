import cv2
import mediapipe as mp
import pickle
import numpy as np
import speech_recognition as sr
import pyttsx3
import threading

recognizer = sr.Recognizer()
engine = pyttsx3.init(driverName='sapi5')
voices = engine.getProperty('voices')
selected_voice = voices[0]
engine.setProperty('voice', selected_voice.id)
engine.setProperty('rate', 140)

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'Hey!', 1: 'Shhhh!', 2: 'Rock'}

prev_prediction = None
predicted_words = []

# Initialize flag to track if TTS engine is running
tts_engine_running = False

def speak(text):
    global tts_engine_running
    # Ensure only one thread starts the TTS engine
    if not tts_engine_running:
        tts_engine_running = True
        engine.say(text)
        engine.runAndWait()
        tts_engine_running = False

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
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x)
                data_aux.append(y)
                x_.append(x)
                y_.append(y)
        
        x1 = int(min(x_) * W)
        y1 = int(min(y_) * H)
        x2 = int(max(x_) * W)
        y2 = int(max(y_) * H)

        max_length = 84
        padded_data_aux = np.pad(data_aux, (0, max_length - len(data_aux)), mode='constant')
        padded_data_aux = padded_data_aux.reshape(1, -1)
        prediction = model.predict(padded_data_aux)
        predicted_char = labels_dict[int(prediction[0])]

        if predicted_char != prev_prediction:
            print(predicted_char)
            predicted_words.append(predicted_char)
            threading.Thread(target=speak, args=(predicted_char,)).start()
            prev_prediction = predicted_char
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_char, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()