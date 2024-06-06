import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Function to load metadata and predictions
def load_metadata_and_predictions(metadata_file, predictions_file):
    metadata_df = pd.read_csv(metadata_file)
    predictions_df = pd.read_csv(predictions_file)

    # Merge the dataframes on the 'Filename' column
    merged_df = pd.merge(metadata_df, predictions_df, on='Filename')

    # Convert 'Category' to binary (0 for Clear, 1 for Cloud)
    merged_df['Category'] = merged_df['Category'].apply(lambda x: 0 if x == 'Clear' else 1)

    return merged_df

# Load the metadata and predictions for the model
base_dir = 'SeparatedImageDataMay30'
metadata_file = os.path.join(base_dir, '00637_metadata.csv')
predictions_file = os.path.join(base_dir, 'predictions_dl_00637.csv')
merged_df = load_metadata_and_predictions(metadata_file, predictions_file)

# Extract the time step from the filename
merged_df['Time Step'] = merged_df['Filename'].str.extract(r'_time_step_(\d+)\.png')[0].astype(int)

# Sort the dataframe by 'Time Step'
merged_df = merged_df.sort_values('Time Step')

# Calculate accuracy and confusion matrix for the model
true_labels = merged_df['Category']
predicted_labels = merged_df['Predicted Label']

accuracy = accuracy_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print(f"Accuracy (Model - DL): {accuracy}")
print("Confusion Matrix (Model - DL):")
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Clear', 'Cloud'])
disp.plot()
plt.title("Confusion Matrix (Model - DL)")
plt.show()

# Apply running average to predicted probabilities
window_size = 10  # Define the window size for the running average
merged_df['Running Avg Probability'] = merged_df['Predicted Probability'].rolling(window=window_size).mean()

# Drop NaN values resulting from the rolling operation
merged_df = merged_df.dropna(subset=['Running Avg Probability']).reset_index(drop=True)

# Generate new predictions based on the running average probabilities
merged_df['Smoothed Predicted Label'] = merged_df['Running Avg Probability'].apply(lambda x: 1 if x >= 0.5 else 0)

# Calculate accuracy and confusion matrix for the smoothed predictions
smoothed_true_labels = true_labels[window_size-1:].reset_index(drop=True)
smoothed_accuracy = accuracy_score(smoothed_true_labels, merged_df['Smoothed Predicted Label'])
smoothed_conf_matrix = confusion_matrix(smoothed_true_labels, merged_df['Smoothed Predicted Label'])

print(f"Smoothed Accuracy (Model - DL): {smoothed_accuracy}")
print("Smoothed Confusion Matrix (Model - DL):")
disp_smoothed = ConfusionMatrixDisplay(smoothed_conf_matrix, display_labels=['Clear', 'Cloud'])
disp_smoothed.plot()
plt.title("Smoothed Confusion Matrix (Model - DL)")
plt.show()

# Plotting
plt.figure(figsize=(15, 6))

# Plot true labels
plt.plot(merged_df['Time Step'], smoothed_true_labels, 'ko', label='True Label (Category)', markersize=5)

# Plot predicted probabilities for the model
plt.plot(merged_df['Time Step'], merged_df['Predicted Probability'], 'g', label='Predicted Probability (Model - DL)', linewidth=2, alpha=0.7)

# Plot running average of predicted probabilities for the model
plt.plot(merged_df['Time Step'], merged_df['Running Avg Probability'], 'r', label='Running Avg Probability (Model - DL)', linewidth=2, alpha=0.7)

plt.axhline(y=0.5, color='k', linestyle='--', label='50% Probability')
plt.xlabel('Time Step')
plt.ylabel('Label/Probability')
plt.title('Time Step vs Predicted Probability and True Label')
plt.legend()
plt.show()
