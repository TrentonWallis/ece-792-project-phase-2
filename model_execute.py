import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses, Model
from random import randint
import numpy as np
import pickle

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

filestring = '_8_4_2_16.dat'
[x_train,y_train] = load_pickle('train'+filestring)
[x_valid,y_valid] = load_pickle('valid'+filestring)
[x_test,y_test] = load_pickle('test'+filestring)

plt.figure()
plt.subplot(121)
plt.imshow(np.array(y_train[2][1]).astype(np.uint8), interpolation='none')
plt.subplot(122)
plt.imshow(np.array(x_train[2][1]).astype(np.uint8), interpolation='none')
plt.show()
# x_train = np.array(x_train)
# x_valid = np.array(x_valid)
# x_test = np.array(x_test)

# x_train, x_valid, x_test = x_train / 255.0 ,x_valid / 255.0, x_test / 255.0
# print(x_valid.shape)

# decoder = models.Sequential()
# decoder.add(layers.Conv2D(128, 3, strides=1, padding='same', activation='relu', input_shape=encoder.output.shape[1:]))
# decoder.add(layers.UpSampling2D(2))
# decoder.add(layers.Conv2D(16, 3, strides=1, padding='same', activation='relu'))
# decoder.add(layers.UpSampling2D(2))
# decoder.add(layers.Conv2D(3, 3, strides=1, padding='same', activation='relu'))
# decoder.add(layers.UpSampling2D(2))
# decoder.summary()


# CNNDecoder = Model(inputs=decoder.input, outputs=decoder.outputs)
# CNNDecoder.compile(optimizer='adam', loss=losses.mean_squared_error)
# history = CNNDecoder.fit(x_train, x_train, batch_size=64, epochs=40, validation_data=(x_test, x_test))


# fig, axs = plt.subplots(figsize=(15,15))

# axs.plot(history.history['loss'])
# axs.plot(history.history['val_loss'])
# axs.title.set_text('Training Loss vs Validation Loss')
# axs.set_xlabel('Epochs')
# axs.set_ylabel('Loss')
# axs.legend(['Train','Val'])