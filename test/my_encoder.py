import gc

import matplotlib.pyplot as plt
import numpy as np
from keras import Input
from keras.layers import MaxPooling2D, UpSampling2D, Conv2D
from keras import layers
from keras.models import Model
import os
from PIL import Image
from numpy import array

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

sonar_train = sonar_train.astype('float32') / 255.
sonar_test = sonar_test.astype('float32') / 255.
sonar_test_noisy = sonar_test_noisy.astype('float32') / 255.
sonar_train_noisy = sonar_train_noisy.astype('float32') / 255.

# n = 1
plt.figure()
# display original images
plt.axis('off')
plt.imshow(sonar_test[2])
plt.show()

plt.figure()
plt.axis('off')
plt.imshow(sonar_test_noisy[2])
plt.show()
# plt.gray()
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)

# display noise images
# ax = plt.subplot(2, n, i + 1 + n)
# plt.imshow(sonar_test_noisy[2])
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# plt.show()

# 输入层
input_img = Input(shape=(256, 256, 3))
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)  # (?, 28, 28, 32)
x = MaxPooling2D((2, 2), padding='same')(x)  # (?, 14, 14, 32)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # (?, 14, 14, 32)
encoded = MaxPooling2D((2, 2), padding='same')(x)  # (?, 7, 7, 32)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)  # (?, 7, 7, 32)
x = UpSampling2D((2, 2))(x)  # (?, 14, 14, 32)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)  # (?, 14, 14, 32)
x = UpSampling2D((2, 2))(x)  # (?, 28, 28, 32)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)  # (?, 28, 28, 1)

auto_encoder = Model(input_img, decoded)
auto_encoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# auto_encoder.compile(optimizer='sgd', loss='binary_crossentropy')
#
auto_encoder.fit(sonar_train_noisy, sonar_train,  # 输入输出
                 epochs=10,  # 迭代次数
                 batch_size=10,
                 shuffle=True,
                 validation_data=(sonar_test_noisy, sonar_test))

decoded_imgs = auto_encoder.predict(sonar_test_noisy)  # 测试集合输入查看器去噪之后输出。

# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow((sonar_test_noisy[i].reshape(256, 256, 3)*255).astype(np.uint8))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow((decoded_imgs[i].reshape(256, 256, 3) * 255).astype(np.uint8))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

plt.figure()
plt.axis('off')
plt.imshow((decoded_imgs[2].reshape(256, 256, 3) * 255).astype(np.uint8))
plt.show()
