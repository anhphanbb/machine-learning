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

# Load the metadata and predictions for the model and transfer learning
base_dir = 'SeparatedImageDataMay30'
metadata_file = os.path.join(base_dir, '00636_metadata.csv')
predictions_file_dl = os.path.join(base_dir, 'predictions_dl_00636.csv')
predictions_file_tl = os.path.join(base_dir, 'predictions_tl_00636.csv')

merged_df_dl = load_metadata_and_predictions(metadata_file, predictions_file_dl)
merged_df_tl = load_metadata_and_predictions(metadata_file, predictions_file_tl)

# Extract the time step from the filename
merged_df_dl['Time Step'] = merged_df_dl['Filename'].str.extract(r'_time_step_(\d+)\.png')[0].astype(int)
merged_df_tl['Time Step'] = merged_df_tl['Filename'].str.extract(r'_time_step_(\d+)\.png')[0].astype(int)

# Sort the dataframes by 'Time Step'
merged_df_dl = merged_df_dl.sort_values('Time Step')
merged_df_tl = merged_df_tl.sort_values('Time Step')

# Calculate accuracy and confusion matrix for the model and transfer learning
true_labels = merged_df_dl['Category']
predicted_labels_dl = merged_df_dl['Predicted Label']
predicted_labels_tl = merged_df_tl['Predicted Label']

accuracy_dl = accuracy_score(true_labels, predicted_labels_dl)
conf_matrix_dl = confusion_matrix(true_labels, predicted_labels_dl)

accuracy_tl = accuracy_score(true_labels, predicted_labels_tl)
conf_matrix_tl = confusion_matrix(true_labels, predicted_labels_tl)

print(f"Accuracy (Model - DL): {accuracy_dl}")
print("Confusion Matrix (Model - DL):")
disp_dl = ConfusionMatrixDisplay(conf_matrix_dl, display_labels=['Clear', 'Cloud'])
disp_dl.plot()
plt.title("Confusion Matrix (Model - DL)")
plt.show()

print(f"Accuracy (Model - TL): {accuracy_tl}")
print("Confusion Matrix (Model - TL):")
disp_tl = ConfusionMatrixDisplay(conf_matrix_tl, display_labels=['Clear', 'Cloud'])
disp_tl.plot()
plt.title("Confusion Matrix (Model - TL)")
plt.show()

# Apply running average to predicted probabilities
window_size = 10  # Define the window size for the running average
merged_df_dl['Running Avg Probability'] = merged_df_dl['Predicted Probability'].rolling(window=window_size).mean()
merged_df_tl['Running Avg Probability'] = merged_df_tl['Predicted Probability'].rolling(window=window_size).mean()

# Drop NaN values resulting from the rolling operation
merged_df_dl = merged_df_dl.dropna(subset=['Running Avg Probability']).reset_index(drop=True)
merged_df_tl = merged_df_tl.dropna(subset=['Running Avg Probability']).reset_index(drop=True)

# Generate new predictions based on the running average probabilities
merged_df_dl['Smoothed Predicted Label'] = merged_df_dl['Running Avg Probability'].apply(lambda x: 1 if x >= 0.5 else 0)
merged_df_tl['Smoothed Predicted Label'] = merged_df_tl['Running Avg Probability'].apply(lambda x: 1 if x >= 0.5 else 0)

# Calculate accuracy and confusion matrix for the smoothed predictions
smoothed_true_labels = true_labels[window_size-1:].reset_index(drop=True)
smoothed_accuracy_dl = accuracy_score(smoothed_true_labels, merged_df_dl['Smoothed Predicted Label'])
smoothed_conf_matrix_dl = confusion_matrix(smoothed_true_labels, merged_df_dl['Smoothed Predicted Label'])

smoothed_accuracy_tl = accuracy_score(smoothed_true_labels, merged_df_tl['Smoothed Predicted Label'])
smoothed_conf_matrix_tl = confusion_matrix(smoothed_true_labels, merged_df_tl['Smoothed Predicted Label'])

print(f"Smoothed Accuracy (Model - DL): {smoothed_accuracy_dl}")
print("Smoothed Confusion Matrix (Model - DL):")
disp_smoothed_dl = ConfusionMatrixDisplay(smoothed_conf_matrix_dl, display_labels=['Clear', 'Cloud'])
disp_smoothed_dl.plot()
plt.title("Smoothed Confusion Matrix (Model - DL)")
plt.show()

print(f"Smoothed Accuracy (Model - TL): {smoothed_accuracy_tl}")
print("Smoothed Confusion Matrix (Model - TL):")
disp_smoothed_tl = ConfusionMatrixDisplay(smoothed_conf_matrix_tl, display_labels=['Clear', 'Cloud'])
disp_smoothed_tl.plot()
plt.title("Smoothed Confusion Matrix (Model - TL)")
plt.show()

# Plotting
plt.figure(figsize=(15, 6))

# Plot true labels
plt.plot(merged_df_dl['Time Step'], smoothed_true_labels, 'ko', label='True Label (Category)', markersize=5)

# Plot predicted probabilities for the model and transfer learning
plt.plot(merged_df_dl['Time Step'], merged_df_dl['Predicted Probability'], 'g', label='Predicted Probability (Model - DL)', linewidth=2, alpha=0.7)
plt.plot(merged_df_tl['Time Step'], merged_df_tl['Predicted Probability'], 'b', label='Predicted Probability (Model - TL)', linewidth=2, alpha=0.7)

# Plot running average of predicted probabilities for the model and transfer learning
plt.plot(merged_df_dl['Time Step'], merged_df_dl['Running Avg Probability'], 'r', label='Running Avg Probability (Model - DL)', linewidth=2, alpha=0.7)
plt.plot(merged_df_tl['Time Step'], merged_df_tl['Running Avg Probability'], 'c', label='Running Avg Probability (Model - TL)', linewidth=2, alpha=0.7)

plt.axhline(y=0.5, color='k', linestyle='--', label='50% Probability')
plt.xlabel('Time Step')
plt.ylabel('Label/Probability')
plt.title('Time Step vs Predicted Probability and True Label')
plt.legend()
plt.show()
