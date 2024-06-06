import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2

# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# 3. Load Data as Grayscale and Resize
data = tf.keras.utils.image_dataset_from_directory(
    'ImageDataMay30',
    image_size=(300, 300),
    color_mode='grayscale'
)

data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()

# Visualize the first few images
fig, ax = plt.subplots(ncols=4, figsize=(20,20))
for idx, img in enumerate(batch[0][:4]):
    ax[idx].imshow(img.astype(int))
    ax[idx].title.set_text(batch[1][idx])

# 4. Scale Data
data = data.map(lambda x,y: (x/255, y))

# 5. Split Data
train_size = int(len(data)*.7)
val_size = int(len(data)*.2)
test_size = int(len(data)*.1)

train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

from tensorflow.keras.callbacks import EarlyStopping

# 6. Build Deep Learning Model
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(300,300,1)),
    MaxPooling2D(),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(),
    Conv2D(16, (3,3), activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 7. Train
logdir='logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
early_stopping_callback = EarlyStopping(
    monitor='val_loss',   # You can also monitor 'val_accuracy'
    patience=10,           # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restores model weights from the epoch with the best value of the monitored quantity
)

hist = model.fit(
    train,
    epochs=45,
    validation_data=val,
    callbacks=[tensorboard_callback, early_stopping_callback]
)

# 8. Plot Performance
fig = plt.figure()
plt.plot(hist.history['loss'], label='loss')
plt.plot(hist.history['val_loss'], label='val_loss')
plt.title('Loss Trends')
plt.legend()
plt.show()

fig = plt.figure()
plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['val_accuracy'], label='val_accuracy')
plt.title('Accuracy Trends')
plt.legend()
plt.show()

# 9. Evaluate
pre, re, acc = tf.metrics.Precision(), tf.metrics.Recall(), tf.metrics.BinaryAccuracy()
for batch in test.as_numpy_iterator():
    X, y = batch
    yhat = model.predict(X)
    pre.update_state(y, yhat)
    re.update_state(y, yhat)
    acc.update_state(y, yhat)

print(f"Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}")

# 11. Save the Model
model.save('models/DeepLearning.h5')
new_model = load_model('models/DeepLearning.h5')
