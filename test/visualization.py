from keras.applications import VGG16
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing import image

base_dir = 'D:\kaggleDogVSCatData\cats_and_dogs_small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
train_dir = os.path.join(base_dir, 'train')
test_cats_dir = os.path.join(train_dir, 'cats')
img_path = os.path.join(test_cats_dir, 'cat.112.jpg')
img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)
# plt.imshow(img_tensor[0])
# plt.show()

model = models.load_model('cats_and_dogs_small_1.h5')
model.summary()
# conv_base = VGG16(include_top=False, input_shape=(150, 150, 3))
# model = models.Sequential()
# model.add(conv_base)
# # 以vgg16作为卷积基
# model.add(layers.Flatten())
# model.add(layers.Dense(256, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))
# model.load_weights('model_weights.h5')
layer_output = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_output)
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[7]
plt.matshow(first_layer_activation[0, :, :, 101])
plt.show()
