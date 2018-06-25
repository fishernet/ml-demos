import keras
import random
import matplotlib.pyplot as plt

from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

print('keras : version {}'.format(keras.__version__))

# Download the dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

# Let's have a look at the training data:
train_images.shape
train_labels

# Uncomment to see an image
# i = random.randint(0, 100)
# plt.imshow(train_images[i], cmap=plt.cm.binary)
# plt.show()

# Prepare your data
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# We will now compile and print out a summary of our model
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

# Train the model
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Test
loss, accuracy = model.evaluate(test_images, test_labels)
print('Test accuracy: %.2f' % (accuracy))
print('Test loss: %.2f' % (loss))
