import pandas as pd
import tensorflow as tf
import os
import cv2
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import numpy as np

# Reload the model
new_model = load_model('models/DeepLearning_resnet_model.h5')

# Manually Create a Dataset from 'SeparatedImageDataMay30' Directory
base_dir = 'SeparatedImageDataMay30'
subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

# Function to undo preprocessing
def undo_preprocessing(img):
    img = img.copy()
    img[..., 0] += 123.68
    img[..., 1] += 116.779
    img[..., 2] += 103.939
    return np.clip(img, 0, 255).astype('uint8')

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
                img = cv2.imread(filepath)
                img = cv2.resize(img, (300, 300))
                img_preprocessed = np.expand_dims(img, axis=0)
                img_preprocessed = tf.keras.applications.resnet50.preprocess_input(img_preprocessed)
                images.append(img_preprocessed)
                image_paths.append(filename)
                
                if not subfolder_metadata.empty:
                    file_metadata = subfolder_metadata[subfolder_metadata['Filename'] == filename]
                    if not file_metadata.empty:
                        metadata.append(file_metadata)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    if len(images) == 0:
        print(f"No valid images found in {subfolder}")
        continue

    images = np.vstack(images)
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
    #print(results_df.head())
    
    # if not metadata_df.empty:
    #     results_df = pd.merge(results_df, metadata_df, on='Filename', how='left')

    output_file = os.path.join(base_dir, f'predictions_tl_{subfolder}.csv')
    results_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")

    # Visualize some predictions
    fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
    for i, ax in enumerate(axes.flatten()):
        if i < len(images):
            display_img = undo_preprocessing(images[i].squeeze())
            ax.imshow(display_img)
            ax.set_title(f"Pred: {predictions[i]}, Prob: {probabilities[i]:.2f}")
            ax.axis('off')
    plt.tight_layout()
    plt.show()
