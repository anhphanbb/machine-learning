import pandas as pd
import tensorflow as tf
import os
import cv2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from matplotlib import pyplot as plt
import numpy as np


# Reload the model
new_model = load_model('models/DeepLearning.h5')

# 11. Manually Create a Dataset from 'orbit1' Directory
orbit1_dir = 'orbit1'
image_paths = []
images = []

for filename in os.listdir(orbit1_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        filepath = os.path.join(orbit1_dir, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (300, 300))
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=-1)  # Add channel dimension
        images.append(img)
        image_paths.append(filename)

images = np.array(images)

# 12. Make Predictions on All Images in 'orbit1'
predictions = []
probabilities = []

yhat = new_model.predict(images)
predicted_class = (yhat > 0.5).astype(int)

for i in range(len(yhat)):
    predictions.append(predicted_class[i][0])
    probabilities.append(yhat[i][0])

# Save predictions to CSV with the specified structure
results_df = pd.DataFrame({
    'Filename': image_paths,
    'Predicted Label': predictions,
    'Predicted Probability': probabilities
})

results_df.to_csv('orbit1_predictions_dl.csv', index=False)

print("Predictions saved to orbit1_predictions_dl.csv")