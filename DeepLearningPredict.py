import pandas as pd
import tensorflow as tf
import os
import cv2
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np

# Reload the model
new_model = load_model('models/DeepLearning.h5')

# Manually Create a Dataset from 'SeparatedImagesMay30' Directory
base_dir = 'SeparatedImageDataMay30'
subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

# Iterate through subfolders and their corresponding metadata files
for subfolder in subfolders:
    subfolder_path = os.path.join(base_dir, subfolder)
    metadata_file = os.path.join(base_dir, f"{subfolder}_metadata.csv")
    
    image_paths = []
    images = []
    metadata = []

    if os.path.exists(metadata_file):
        subfolder_metadata = pd.read_csv(metadata_file)
    else:
        subfolder_metadata = pd.DataFrame()

    for filename in os.listdir(subfolder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            filepath = os.path.join(subfolder_path, filename)
            try:
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (300, 300))
                img = img / 255.0  # Normalize
                img = np.expand_dims(img, axis=-1)  # Add channel dimension
                images.append(img)
                image_paths.append(filename)
                
                if not subfolder_metadata.empty:
                    file_metadata = subfolder_metadata[subfolder_metadata['Filename'] == filename]
                    metadata.append(file_metadata)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    images = np.array(images)
    metadata_df = pd.concat(metadata, ignore_index=True) if metadata else pd.DataFrame()

    # Make Predictions on All Images in the current subfolder
    predictions = []
    probabilities = []

    # Process images in batches
    batch_size = 32
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size]
        yhat = new_model.predict(batch_images)
        predicted_class = (yhat > 0.5).astype(int)

        for j in range(len(yhat)):
            predictions.append(predicted_class[j][0])
            probabilities.append(yhat[j][0])

    # Save predictions to CSV with the specified structure
    results_df = pd.DataFrame({
        'Filename': image_paths,
        'Predicted Label': predictions,
        'Predicted Probability': probabilities
    })

    #if not metadata_df.empty:
        #results_df = pd.merge(results_df, metadata_df, on='Filename', how='left')


    output_file = os.path.join(base_dir, f'predictions_dl_{subfolder}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Visualize some predictions
    fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            ax.imshow(images[i].squeeze(), cmap='gray')
            ax.set_title(f"Pred: {predictions[i]}, Prob: {probabilities[i]:.2f}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()
