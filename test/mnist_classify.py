from keras import layers
from keras import models
from keras.datasets import mnist
from keras.utils import to_categorical

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPool2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPool2D(2, 2))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

(train_imges, train_labels), (test_imges, test_labels) = mnist.load_data(
    path='D:/workspace/my_keras_test/data/mnist.npz')
train_imges = train_imges.reshape((60000, 28, 28, 1))
train_imges.astype('float32') / 255
test_imges = test_imges.reshape(10000, 28, 28, 1)
test_imges.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_imges, train_labels, epochs=5, batch_size=64)
history_dict = history.history
test_loss, test_acc = model.evaluate(test_imges, test_labels)
print(test_acc)
