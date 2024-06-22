from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import io
from PIL import Image
import numpy as np
import cv2
from  ultralytics import YOLO

# Define your labels list here so it's accessible throughout the script
labels = ['baby', 'eat', 'father', 'finish', 'happy', 'house', 'important', 'love', 'mall', 'me', 'mosque', 'mother', 'normal', 'sad', 'stop', 'thanks', 'worry']

app = Flask(__name__)

model = YOLO('best.pt')

# Function to process a single frame and return the predicted word
def predict_from_frame(frame):
    results = model(frame)
    if results and hasattr(results[0], 'probs'):
        sign = results[0].probs.top1
        word = labels[sign]
    else:
        word = "Prediction unavailable"
    print (word)
    return word

# Generator function to capture frames from the camera
def generate_frames(frame):
    word = predict_from_frame(frame)
    print(f"Predicted word: {word}")  # Print the predicted word
    return word.encode()



@app.route('/upload', methods=['POST'])
def upload_image():
    if 'frame' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['frame']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save the image in memory
    in_memory_file =in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    in_memory_file.seek(0)  # Move the pointer to the beginning of the file
    
    # Open the image with PIL
    pil_image = Image.open(in_memory_file)

    # Convert the PIL image to a NumPy array
    frame = np.array(pil_image)
    output = generate_frames(frame)
    return output

if __name__ == '__main__':
    app.run(debug=True)