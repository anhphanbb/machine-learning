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

# Load the metadata and predictions for the first model
parent_folder_path = os.path.dirname('./SeparatedImageDataMay30')
orbit1_metadata_file = os.path.join(parent_folder_path, 'orbit1_metadata.csv')
orbit1_predictions_file = os.path.join(parent_folder_path, 'orbit1_predictions.csv')
merged_df = load_metadata_and_predictions(orbit1_metadata_file, orbit1_predictions_file)

# Extract the time step from the filename
merged_df['Time Step'] = merged_df['Filename'].str.extract(r'_time_step_(\d+)\.png')[0].astype(int)

# Sort the dataframe by 'Time Step'
merged_df = merged_df.sort_values('Time Step')

# Calculate accuracy and confusion matrix for the first model
true_labels = merged_df['Category']
predicted_labels = merged_df['Predicted Label']

accuracy = accuracy_score(true_labels, predicted_labels)
conf_matrix = confusion_matrix(true_labels, predicted_labels)

print(f"Accuracy (Model 1): {accuracy}")
print("Confusion Matrix (Model 1):")
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Clear', 'Cloud'])
disp.plot()
plt.title("Confusion Matrix (Model 1)")
plt.show()

# Load the metadata and predictions for the second model
orbit1_predictions_file_svm = os.path.join(parent_folder_path, 'orbit1_predictions_svm.csv')
merged_df_svm = load_metadata_and_predictions(orbit1_metadata_file, orbit1_predictions_file_svm)

# Extract the time step from the filename for the second model
merged_df_svm['Time Step'] = merged_df_svm['Filename'].str.extract(r'_time_step_(\d+)\.png')[0].astype(int)

# Sort the dataframe by 'Time Step' for the second model
merged_df_svm = merged_df_svm.sort_values('Time Step')

# Calculate accuracy and confusion matrix for the second model
predicted_labels_svm = merged_df_svm['Predicted Label']

accuracy_svm = accuracy_score(true_labels, predicted_labels_svm)
conf_matrix_svm = confusion_matrix(true_labels, predicted_labels_svm)

print(f"Accuracy (Model 2 - SVM): {accuracy_svm}")
print("Confusion Matrix (Model 2 - SVM):")
disp_svm = ConfusionMatrixDisplay(conf_matrix_svm, display_labels=['Clear', 'Cloud'])
disp_svm.plot()
plt.title("Confusion Matrix (Model 2)")
plt.show()

# Load the metadata and predictions for the deep learning model
orbit1_predictions_file_dl = os.path.join(parent_folder_path, 'orbit1_predictions_dl.csv')
merged_df_dl = load_metadata_and_predictions(orbit1_metadata_file, orbit1_predictions_file_dl)

# Extract the time step from the filename for the deep learning model
merged_df_dl['Time Step'] = merged_df_dl['Filename'].str.extract(r'_time_step_(\d+)\.png')[0].astype(int)

# Sort the dataframe by 'Time Step' for the deep learning model
merged_df_dl = merged_df_dl.sort_values('Time Step')

# Calculate accuracy and confusion matrix for the deep learning model
predicted_labels_dl = merged_df_dl['Predicted Label']

accuracy_dl = accuracy_score(true_labels, predicted_labels_dl)
conf_matrix_dl = confusion_matrix(true_labels, predicted_labels_dl)

print(f"Accuracy (Model 3 - DL): {accuracy_dl}")
print("Confusion Matrix (Model 3 - DL):")
disp_dl = ConfusionMatrixDisplay(conf_matrix_dl, display_labels=['Clear', 'Cloud'])
disp_dl.plot()
plt.title("Confusion Matrix (Model 3 - DL)")
plt.show()

# Plotting
plt.figure(figsize=(15, 6))

# Plot true labels
plt.plot(merged_df['Time Step'], merged_df['Category'], 'ko', label='True Label (Category)', markersize=5)

# Plot predicted probabilities for the first model
plt.plot(merged_df['Time Step'], merged_df['Predicted Probability'], 'r', label='Predicted Probability (Model 1 - LR)', linewidth=2, alpha=0.7)

# Plot predicted probabilities for the second model
plt.plot(merged_df_svm['Time Step'], merged_df_svm['Predicted Probability'], 'b', label='Predicted Probability (Model 2 - SVM)', linewidth=2, alpha=0.7)

# Plot predicted probabilities for the deep learning model
plt.plot(merged_df_dl['Time Step'], merged_df_dl['Predicted Probability'], 'g', label='Predicted Probability (Model 3 - DL)', linewidth=2, alpha=0.7)

plt.axhline(y=0.5, color='k', linestyle='--', label='50% Probability')
plt.xlabel('Time Step')
plt.ylabel('Label/Probability')
plt.title('Time Step vs Predicted Probability and True Label')
plt.legend()
plt.show()
