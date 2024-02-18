import os
import pickle

import mediapipe as mp
import cv2
import matplotlib.pyplot as plt


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode = True, min_detection_confidence = 0.3)

DATA_DIR = './data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR): # for all dirs
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)): # for each image in each dir
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))  # convert image to cv2
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # then convert to rgb so you can use matplotlib to plot
        
        results = hands.process(img_rgb) # using mediapipe to add landmarks to each hand
        if results.multi_hand_landmarks: # checks to ensure atleast one hand has been detected
            for hand_landmarks in results.multi_hand_landmarks: # for each img_path (images) we are extracting all the landmarks
                
                # creating an array of all the landmarks we actually need
                for i in range(len(hand_landmarks.landmark)): 
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x) # then creating an array with all the x and y values of the landmarks
                    data_aux.append(y)
                    
            data.append(data_aux) # creating an entire list of all these arrays
            labels.append(dir_) # then the name of the directory of all the images, creating the dataset we need to train the classifier
            
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
print('Done')