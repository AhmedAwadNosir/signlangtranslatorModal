import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import seaborn as sns
import matplotlib.image as img
import matplotlib.pyplot as plt
import json

# Define your labels and dictionary here
labels = ['baby', 'eat', 'father', 'finish', 'happy', 'house', 'important', 'love', 'mall', 'me', 'mosque', 'mother', 'normal', 'sad', 'stop', 'thanks', 'worry']
word_dic = {'baby': 'طفل',
            'eat': 'يأكل',
            # ... (other sign words in Arabic)
            'worry': 'يقلق'}

# Assuming you have already trained and validated your model and have the best.pt file ready

from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO(r'E:\modal\best.pt')

# Define path to the video or use 0 for webcam
Video_path = 0  # or specify the path to your video file

# Initialize a variable to store the sign word
predicted_word = ""

# Real-time sign language recognition
import cv2

cap = cv2.VideoCapture(Video_path)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    img = cv2.resize(frame, (128, 128))
    results = model(img)
    for r in results:
        sign = r.probs.top1
        predicted_word = labels[sign]
        # Uncomment the line below to display the Arabic sign word
        # predicted_word = word_dic[labels[sign]]
    cv2.putText(frame, predicted_word, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('sign', frame)
    if cv2.waitKey(1) == ord('q'):
        break

print

cap.release()
cv2.destroyAllWindows()

# Save the predicted sign word to a JSON file
output_json = {"predicted_word": predicted_word}
with open("predicted_sign_word.json", 'w') as json_file:
    json.dump(output_json, json_file)

print(f"Predicted sign word: {predicted_word}")
print(f"Saved to predicted_sign_word.json")
