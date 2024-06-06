# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 14:38:01 2024

@author: Anh
"""

import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Plot the model
plot_model(model, to_file='cnn_model.png', show_shapes=True, show_layer_names=True)

# Display the plot
img = plt.imread('cnn_model.png')
plt.imshow(img)
plt.axis('off')
plt.show()
