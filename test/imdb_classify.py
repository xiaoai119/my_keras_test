
import matplotlib.pyplot as plt
from keras.datasets import imdb
import numpy as np
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics

# (train_imge, train_labels), (test_imge, test_labels) = mnist.load_data(path='D:/workspace/my_keras_test/data/mnist.npz')
# model = Sequential()
# model.add(Dense(units=64, input_dim=100))
# model.add(Activation("relu"))
# model.add(Dense(units=10))
# model.add(Activation("softmax"))
# model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path='D:/workspace/my_keras_test/data/imdb.npz',
                                                                      num_words=10000)
word_index = imdb.get_word_index(path='D:/workspace/my_keras_test/data/imdb_word_index.json')
reverse_word_index = dict((value, key) for (key, value) in word_index.items())
decoded_review = ''.join(reverse_word_index.get(i - 3, '?') for i in train_data[0])


# print(decoded_review)
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequences in enumerate(sequences):
        results[i, sequences] = 1
    return results


# 转矩阵
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
# x_train = np.asarray(train_data).astype('float32')
# x_test = np.asarray(test_data).astype('float32')
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

model = models.Sequential()
model.add(layers.Dense(units=16, activation='relu', input_shape=(10000,)))
# 添加随机失活
model.add(layers.Dropout(0.5))
# input shape 为 tuple 类型
# input shape only the first layer
model.add(layers.Dense(units=16, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(units=1, activation='sigmoid'))
model.compile(optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=['acc'])
# 改函数式
x_val = x_train[:10000, :]
partial_x_train = x_train[10000:, :]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
history = model.fit(partial_x_train, partial_y_train, epochs=10, batch_size=500, validation_data=(x_val, y_val))


def plt_train_loss():
    history_dict = history.history
    loss_value = history_dict['loss']
    val_loss_value = history_dict['val_loss']
    epochs = range(1, len(val_loss_value) + 1)
    plt.plot(epochs, loss_value, 'ro', label='Training loss')
    plt.plot(epochs, val_loss_value, 'b', label='Validation loss')
    plt.title('loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


