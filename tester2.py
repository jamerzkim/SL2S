import cv2
import socket
import pickle
import os
import numpy as np

# Import your model and any necessary dependencies
import mediapipe as mp
import pickle

# Initialize the hands module from mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Load your trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']
labels_dict = {0: 'Hi!', 1: 'How', 2: 'are you', 3: 'today?', 4: 'i love you', 5: 'Quiet'}

# Create a UDP socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)

# Set the server IP and port
server_ip = "10.232.43.130"
server_port = 6666

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while cap.isOpened():
    # Read a frame from the webcam
    ret, img = cap.read()

    # Display the frame
    cv2.imshow('Img Client', img)

    # Perform prediction using your model
    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Process the frame with the hands module
    results = hands.process(frame_rgb)
    # Check if hand landmarks are detected
    if results.multi_hand_landmarks:
        # Extract hand landmarks
        for hand_landmarks in results.multi_hand_landmarks:
            # Process the hand landmarks for prediction
            # You can modify this part according to your model input requirements
            # Here, we're just extracting x and y coordinates of hand landmarks
            data_aux = []
            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x)
                data_aux.append(landmark.y)
            # Prepare the data for prediction
            required_feature_size = model.n_features_in_

        # Pad the data with zeros based on the required feature size
            padded_data_aux = np.pad(data_aux, (0, required_feature_size - len(data_aux)), mode='constant')
            padded_data_aux = padded_data_aux.reshape(1, -1)
            prediction = model.predict(padded_data_aux)
            predicted_char = labels_dict[int(prediction[0])]
            # Display the predicted label (you can customize how to display it)
            cv2.putText(img, predicted_char, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Encode the frame as JPEG
    ret, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 30])

    # Serialize the frame and send it over UDP to the server
    x_as_bytes = pickle.dumps(buffer)
    s.sendto(x_as_bytes, (server_ip, server_port))

    # Check for the 'Esc' key to exit the loop
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()