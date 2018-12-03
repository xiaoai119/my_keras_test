import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras import Input
from keras.callbacks import TensorBoard
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D, UpSampling2D
from keras.models import Model
from numpy import array

input_img = Input(shape=(256, 256, 3))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, 3, 3, activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# (x_train, _), (x_test, _) = mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
# x_train = np.reshape(x_train, (len(x_train), 1, 28, 28))
# x_test = np.reshape(x_test, (len(x_test), 1, 28, 28))

train_list = [os.path.join('D:\workspace\my_keras_test\sonar_train', f) for f in
              os.listdir('D:\workspace\my_keras_test\sonar_train')]
test_list = [os.path.join('D:\workspace\my_keras_test\sonar_test', f) for f in
             os.listdir('D:\workspace\my_keras_test\sonar_test')]
sonar_train = []
sonar_test = []
for img in train_list:
    img_array = array(Image.open(img))
    sonar_train.append(img_array)
sonar_train = np.array(sonar_train)
sonar_train = np.reshape(sonar_train, (len(sonar_train), 256, 256, 3))
for img in test_list:
    img_array = array(Image.open(img))
    sonar_test.append(img_array)
sonar_test = np.array(sonar_test)
sonar_test = np.reshape(sonar_test, (len(sonar_test), 256, 256, 3))
noise_factor = 0.5
# sonar_train_noisy = sonar_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=sonar_train.shape)
sonar_train_noisy = sonar_train + noise_factor * np.random.normal(loc=0.0, scale=1, size=sonar_train.shape).astype(
    np.uint8)
sonar_test_noisy = sonar_test + noise_factor * np.random.normal(loc=0.0, scale=1, size=sonar_test.shape).astype(
    np.uint8)

# sonar_train_noisy = np.clip(sonar_train_noisy, 0., 1.)
sonar_test_noisy = np.clip(sonar_test_noisy, 0., 256.).astype(np.uint8)
sonar_train_noisy = np.clip(sonar_train_noisy, 0., 256.).astype(np.uint8)

autoencoder.fit(sonar_train_noisy, sonar_train,
                nb_epoch=50,
                batch_size=128,
                shuffle=True,
                validation_data=(sonar_test_noisy, sonar_test))

decoded_imgs = autoencoder.predict(sonar_test)

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i)
    plt.imshow(sonar_test[i].reshape(256, 256, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(256, 256, 3))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
