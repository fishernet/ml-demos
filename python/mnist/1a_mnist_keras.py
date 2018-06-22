import keras
import random
import matplotlib.pyplot as plt

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
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

# Encode the data to prep it for the model
train_labels = keras.utils.to_categorical(train_labels)
test_labels = keras.utils.to_categorical(test_labels)

# Build the model
model = keras.Sequential()
model.add(keras.layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(keras.layers.Dense(10, activation='softmax'))

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
