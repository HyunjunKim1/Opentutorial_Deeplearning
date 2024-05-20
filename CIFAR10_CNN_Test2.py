'''
#Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras import backend as K

import matplotlib.pyplot as plt
import numpy as np

import os

num_classes = 10

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model_path = os.path.join(save_dir, model_name)
model = load_model(model_path)

synset = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
n = 0
plt.imshow(x_test[n].reshape(32, 32, 3), interpolation='nearest')
plt.show()
answer = np.argmax(model.predict(x_test[n].reshape((1, 32, 32, 3))), axis=-1)
print('The Answer is ', synset[answer[0]])

predicted_result = model.predict(x_test)
predicted_labels = np.argmax(predicted_result, axis=1)
test_labels = np.argmax(y_test, axis=1)
wrong_result = []

for n in range(0, len(test_labels)):
    if predicted_labels[n] != test_labels[n]:
        wrong_result.append(n)

count = 0
nrows = 4
ncols = 4
plt.figure(figsize=(12,8))

for n in wrong_result:
    count += 1
    plt.subplot(nrows, ncols, count)
    plt.imshow(x_test[n].reshape(32, 32, 3), cmap='Greys', interpolation='nearest')
    tmp = "Label:" + synset[test_labels[n]] + ", Prediction:" + synset[predicted_labels[n]]
    plt.title(tmp)
    if count == 16:
        break

plt.tight_layout()
plt.show()

