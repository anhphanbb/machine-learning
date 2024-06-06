import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2
import time

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load Data and Resize
data = tf.keras.utils.image_dataset_from_directory(
    'ImageDataMay30',
    image_size=(300, 300),
    color_mode='rgb'  # ResNet50 expects 3 color channels
)

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Visualize the first few images
fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# Preprocess Data for ResNet-50
data = data.map(lambda x, y: (preprocess_input(x), y))

# Split Data
train_size = int(len(data) * .7)
val_size = int(len(data) * .2)
test_size = int(len(data) * .1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

# Define Transfer Learning Model using ResNet-50
def create_resnet_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(300, 300, 3)))
    base_model.trainable = False  # Freeze the base model layers

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

model = create_resnet_model()

# Training and evaluating model
results = []
histories = []

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

logdir = f'logs/resnet_model'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
early_stopping_callback = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

start_time = time.time()

hist = model.fit(
    train,
    epochs=60,
    validation_data=val,
    callbacks=[tensorboard_callback, early_stopping_callback]
)

end_time = time.time()
training_time = end_time - start_time

histories.append(hist)

# Evaluate
pre, re, acc = tf.metrics.Precision(), tf.metrics.Recall(), tf.metrics.BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

precision = pre.result().numpy()
recall = re.result().numpy()
accuracy = acc.result().numpy()

results.append({
    'Model': 'ResNet50',
    'Precision': precision,
    'Recall': recall,
    'Accuracy': accuracy,
    'Training Time (s)': training_time
})

# Save the Model
model.save('models/DeepLearning_resnet_model.h5')

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('resnet_model_performance_comparison.csv', index=False)

print("Performance comparison saved to resnet_model_performance_comparison.csv")

# Save epoch-wise accuracy and validation accuracy
epoch_results = []

for i, hist in enumerate(histories):
    for epoch in range(len(hist.history['accuracy'])):
        epoch_results.append({
            'Model': 'ResNet50',
            'Epoch': epoch + 1,
            'Accuracy': hist.history['accuracy'][epoch],
            'Val_Accuracy': hist.history['val_accuracy'][epoch]
        })

epoch_results_df = pd.DataFrame(epoch_results)
epoch_results_df.to_csv('resnet_epoch_performance_comparison.csv', index=False)

print("Epoch performance comparison saved to resnet_epoch_performance_comparison.csv")

# Plot validation accuracy for ResNet-50 model
plt.figure(figsize=(12, 8))
for i, hist in enumerate(histories):
    plt.plot(hist.history['val_accuracy'], label=f'ResNet50 Val Accuracy')

plt.axhline(y=0.8, color='r', linestyle='--', label='80% Accuracy')
plt.axhline(y=0.9, color='g', linestyle='--', label='90% Accuracy')
plt.title('Validation Accuracy Trends for ResNet-50')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()
