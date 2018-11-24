from keras import Input
import numpy as np
from keras.layers import MaxPooling2D, UpSampling2D, Conv2D
from keras.models import Model
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('minist.nzp', one_hot=True)
x_train, x_test = mnist.train.images, mnist.test.images

x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display noise images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = Input(shape=(28, 28, 1))  # (?, 28, 28, 1)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)  # (?, 28, 28, 32)
x = MaxPooling2D((2, 2), padding='same')(x)  # (?, 14, 14, 32)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # (?, 14, 14, 32)
encoded = MaxPooling2D((2, 2), padding='same')(x)  # (?, 7, 7, 32)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)  # (?, 7, 7, 32)
x = UpSampling2D((2, 2))(x)  # (?, 14, 14, 32)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)  # (?, 14, 14, 32)
x = UpSampling2D((2, 2))(x)  # (?, 28, 28, 32)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)  # (?, 28, 28, 1)

auto_encoder = Model(input_img, decoded)

auto_encoder.compile(optimizer='sgd', loss='mean_squared_error')

auto_encoder.fit(x_train_noisy, x_train,  # 输入输出
                 epochs=1,  # 迭代次数
                 batch_size=128,
                 shuffle=True,
                 validation_data=(x_test_noisy, x_test))

decoded_imgs = auto_encoder.predict(x_test_noisy)  # 测试集合输入查看器去噪之后输出。

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
