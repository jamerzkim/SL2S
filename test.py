import cv2
import mediapipe as mp
import pickle
import numpy as np
import speech_recognition as sr
import pyttsx3
import threading
import socket
import struct

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize speech synthesis engine
engine = pyttsx3.init(driverName='sapi5')
voices = engine.getProperty('voices')
selected_voice = voices[0]
engine.setProperty('voice', selected_voice.id)
engine.setProperty('rate', 140)

# Load the sign language model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: 'Heyyyyy!', 1: 'stop Yapping'}

# Initialize flag to track if TTS engine is running
tts_engine_running = False

# Function to recognize sign language gesture
def recognize_sign_language(frame):
    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
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
        return predicted_char, x1, y1

# Function to send speech data over the network
def send_speech_data(sock, text):
    speech_data = pickle.dumps(text)
    speech_size = struct.pack("L", len(speech_data))
    sock.sendall(speech_size + speech_data)

# Function to receive video frames and speech data over the network
def receive_video_and_speech_data(sock):
    payload_size = struct.calcsize("L")
    while True:
        # Receive video frame
        frame_data = b""
        while len(frame_data) < payload_size:
            frame_data += sock.recv(4096)
        frame_size = struct.unpack("L", frame_data[:payload_size])[0]
        frame_data = frame_data[payload_size:]
        while len(frame_data) < frame_size:
            frame_data += sock.recv(4096)
        frame = pickle.loads(frame_data)
        
        # Recognize sign language gesture
        predicted_char, x1, y1 = recognize_sign_language(frame)

        # Send recognized sign language gesture over the network
        send_speech_data(sock, predicted_char)

        # Draw rectangle and text on the frame
        cv2.rectangle(frame, (x1, y1), (x1 + 100, y1 + 100), (0, 255, 0), 2)
        cv2.putText(frame, predicted_char, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('frame', frame)
        
        # Exit loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Function to establish video call connection
def video_call(address):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.connect(address)
        receive_thread = threading.Thread(target=receive_video_and_speech_data, args=(sock,))
        receive_thread.start()
        receive_thread.join()

# Main function
if __name__ == "__main__":
    address = ('10.232.43.130', )  # Replace with appropriate IP address and port
    video_call(address)