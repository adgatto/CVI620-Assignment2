import os
import cv2
import numpy as np
from joblib import load
import glob

# MODEL
model = load('cat_dog_classifier.joblib')

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))     
    img = img / 255.0                   
    img = img.flatten().reshape(1, -1) 
    return img


def test_on_images(base_folder):
    for subfolder in ['cat', 'dog']:
        folder_path = os.path.join(base_folder, subfolder)
        for image_path in glob.glob(folder_path + '/*.jpg'):  
            img = preprocess_image(image_path)
            prediction = model.predict(img)[0]
            print(f"{image_path} → Predicted: {prediction}")

# RESULTS AND EVALUATION
test_on_images('./test')


# The model only had an accuracy of 60 percent at best. It was not very reliable at differentiating the two animals apart. 