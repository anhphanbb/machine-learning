import pandas as pd
import numpy as np
import os
import cv2
import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from tensorflow.keras.utils import image_dataset_from_directory
import joblib

# Load Data as Grayscale and Resize
data = image_dataset_from_directory(
    'ImageDataMay30',
    image_size=(75, 75),
    color_mode='grayscale'
)

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Preprocess images and labels
images = []
labels = []

for batch in data:
    images.extend(batch[0].numpy())
    labels.extend(batch[1].numpy())

images = np.array(images)
labels = np.array(labels)

# Flatten images for Logistic Regression
images_flattened = images.reshape(images.shape[0], -1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(images_flattened, labels, test_size=0.2, random_state=42)

# Define Logistic Regression model and parameters for grid search
log_reg_model = LogisticRegression(max_iter=10000)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga']
}

# Timing the grid search
start_time = time.time()
grid_search = GridSearchCV(log_reg_model, param_grid, cv=5, verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)
end_time = time.time()
elapsed_time = end_time - start_time

# Best model from grid search
best_model = grid_search.best_estimator_

# Save the best model
joblib.dump(best_model, 'best_log_reg_model.joblib')
print("Best model saved to best_log_reg_model.joblib")

# Evaluate the model
y_pred = best_model.predict(X_test)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print("Best Model Hyperparameters:", grid_search.best_params_)
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)

# Save results
results = {
    'Best Hyperparameters': grid_search.best_params_,
    'Precision': precision,
    'Recall': recall,
    'Accuracy': accuracy,
    'Elapsed Time (s)': elapsed_time
}

results_df = pd.DataFrame([results])
results_df.to_csv('log_reg_model_performance.csv', index=False)

print("Performance saved to log_reg_model_performance.csv")

# Save classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('log_reg_classification_report.csv', index=False)

print("Classification report saved to log_reg_classification_report.csv")

# Save all grid search results, including timing
cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results['fit_time'] = grid_search.cv_results_['mean_fit_time'] + grid_search.cv_results_['mean_score_time']
cv_results.to_csv('log_reg_grid_search_results.csv', index=False)

print("Grid search results saved to log_reg_grid_search_results.csv")

# Prepare data for heatmap
mean_test_scores = grid_search.cv_results_['mean_test_score']
params = grid_search.cv_results_['params']

# Extract values for plotting
C_values = sorted(list(set(param['C'] for param in params)))
penalties = ['l1', 'l2']

heatmap_data = np.zeros((len(C_values), 2))  # Two penalties

for idx, param in enumerate(params):
    C_index = C_values.index(param['C'])
    penalty_index = penalties.index(param['penalty'])
    heatmap_data[C_index, penalty_index] = mean_test_scores[idx]

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, xticklabels=penalties, yticklabels=C_values, cmap='viridis', ax=ax)
ax.set_xlabel('Penalty')
ax.set_ylabel('C')
ax.set_title('Grid Search Accuracy')

plt.show()
