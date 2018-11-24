# import keras
# from keras import layers
# import numpy as np
#
# latent_dim = 32
# height = 32
# width = 32
# channels = 3
#
# generator_input = keras.Input(shape=(latent_dim,))
#
# x = layers.Dense(128 * 16 * 16)(generator_input)
# x = layers.LeakyReLU()(x)
# x = layers.Reshape((16, 16, 128))(x)
#
# x = layers.Conv2D(256, 5, padding='same')(x)
# x = layers.LeakyReLU()(x)
#
# x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
# x = layers.LeakyReLU()(x)
#
# x = layers.Conv2D(256, 5, padding='same')(x)
# x = layers.LeakyReLU()(x)
# x = layers.Conv2D(256, 5, padding='same')(x)
# x = layers.LeakyReLU()(x)
#
# x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
# generator = keras.models.Model(generator_input, x)
# generator.summary()
#
# discriminiator_input = layers.Input(shape=(height, width, channels))
# x = layers.Conv2D(128, 3)(discriminiator_input)
# x = layers.LeakyReLU()(x)
# x = layers.Conv2D(128, 4, strides=2)(x)
# x = layers.LeakyReLU()(x)
# x = layers.Conv2D(128, 4, strides=2)(x)
# x = layers.LeakyReLU()(x)
# x = layers.Conv2D(128, 4, strides=2)(x)
# x = layers.Flatten()(x)
#
# x = layers.Dropout(0.4)(x)
#
# x = layers.Dense(1, activation='sigmoid')(x)
#
# discriminiator = keras.models.Model(discriminiator_input, x)
# discriminiator.summary()
#
# discriminiator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
# discriminiator.compile(optimizer=discriminiator_optimizer, loss='binary_crossentropy')
#
# discriminiator.trainable = False
# gan_input = keras.Input(shape=(latent_dim.))
# gan_output = discriminiator(generator(gan_input))
# gan = keras.models.Model(gan_input, gan_output)
#
# gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
# gan.compile(optimizer=gan_optimizer,loss='binary_crossentropy')
#
#
