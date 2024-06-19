import pandas as pd
import numpy as np
import os
import cv2
import joblib
import matplotlib.pyplot as plt

# Load the best SVM model
best_model = joblib.load('best_svm_model.joblib')

# Function to preprocess images
def preprocess_images(image_dir, img_size=(75, 75)):
    images = []
    filenames = []
    for filename in os.listdir(image_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
            filepath = os.path.join(image_dir, filename)
            try:
                img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, img_size)
                img = img / 255.0  # Normalize
                images.append(img.flatten())  # Flatten the image
                filenames.append(filename)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    return np.array(images), filenames

# Directory containing subfolders with new images
base_dir = 'SeparatedImageDataMay30'
subfolders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]

# Iterate through subfolders and their corresponding metadata files
for subfolder in subfolders:
    subfolder_path = os.path.join(base_dir, subfolder)
    metadata_file = os.path.join(base_dir, f"{subfolder}_metadata.csv")
    
    # Preprocess images in the current subfolder
    new_images, image_filenames = preprocess_images(subfolder_path)
    
    # Make predictions
    if new_images.size > 0:
        predictions = best_model.predict(new_images)
        probabilities = best_model.decision_function(new_images)  # Get decision scores

        # Prepare results DataFrame
        results_df = pd.DataFrame({
            'Filename': image_filenames,
            'Predicted Label': predictions,
            'Decision Score': probabilities
        })
        
        # Merge with metadata if exists
        # if os.path.exists(metadata_file):
        #     metadata_df = pd.read_csv(metadata_file)
        #     results_df = pd.merge(results_df, metadata_df, on='Filename', how='left')
        
        # Save predictions to CSV in the subfolder
        output_file = os.path.join(base_dir, f'predictions_svm_{subfolder}.csv')
        results_df.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")

        # Visualize some predictions
        fig, axes = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
        for i, ax in enumerate(axes.flatten()):
            if i < len(new_images):
                ax.imshow(new_images[i].reshape(75, 75), cmap='gray')
                ax.set_title(f"Pred: {predictions[i]}, Score: {probabilities[i]:.2f}")
                ax.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print(f"No images found in {subfolder_path}")
