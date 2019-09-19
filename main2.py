import os
import zipfile
import pathlib
import random
import pandas as pd
import tensorflow as tf
import sklearn
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import datasets, layers, models
from tensorflow.python.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.python.keras.preprocessing.image import array_to_img, img_to_array, load_img

print(tf.version.VERSION)

AUTOTUNE = tf.data.experimental.AUTOTUNE

## DATAPREP

base_path = pathlib.Path(__file__).parent

base_dir = (base_path / 'data/kagglecatsanddogs_3367a/PetImages').resolve()

cat_dir = (base_path / 'data/kagglecatsanddogs_3367a/PetImages/Cat').resolve()
dog_dir = (base_path / 'data/kagglecatsanddogs_3367a/PetImages/Dog').resolve()

cat_root = pathlib.Path(cat_dir)
print(cat_root)

dog_root = pathlib.Path(dog_dir)
print(dog_root)

data_root = pathlib.Path(base_dir)

cat_image_paths = list(cat_root.glob('*.jpg'))
cat_image_paths = [str(path) for path in cat_image_paths]
# print(cat_image_paths)

dog_image_paths = list(dog_root.glob('*.jpg'))
dog_image_paths = [str(path) for path in dog_image_paths]
# print(cat_image_paths)

all_image_paths = cat_image_paths + dog_image_paths
random.shuffle(all_image_paths)
print(all_image_paths)

image_count = len(all_image_paths)
print(image_count)

label_names = sorted(item.name for item in data_root.glob('*/') if item.is_dir())
print(label_names)
label_to_index = dict((name, index) for index, name in enumerate(label_names))
print(label_to_index)

all_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                    for path in all_image_paths]

print("First 10 labels indices: ", all_image_labels[:10], all_image_paths[:10])

X = all_image_paths
y = all_image_labels
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

print("First 10 test labels indices: ", y_train[:10], x_train[:10])


def preprocess_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [150, 150])
    image /= 255.0  # normalize to [0,1] range

    return image


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)


# The tuples are unpacked into the positional arguments of the mapped function
def load_and_preprocess_from_path_label(path, label):
    print("path:" + path)
    return load_and_preprocess_image(path), label


# print(load_and_preprocess_from_path_label(
#    "C:\\Users\\marti\\PycharmProjects\\untitled4\\data\\kagglecatsanddogs_3367a\\PetImages\\Cat\\1336.jpg",0))

# path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)


train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
print(train_ds)
train_image_label_ds = train_ds.map(load_and_preprocess_from_path_label)
print(train_image_label_ds)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_image_label_ds = val_ds.map(load_and_preprocess_from_path_label)

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_image_label_ds = test_ds.map(load_and_preprocess_from_path_label)

train_fds = train_image_label_ds.shuffle(buffer_size=17500)
#train_fds = train_fds.repeat()
train_fds = train_fds.batch(batch_size=32)
train_fds = train_fds.prefetch(buffer_size=AUTOTUNE)

val_fds = val_image_label_ds.shuffle(buffer_size=3750)
#val_fds = val_fds.repeat()
val_fds = val_fds.batch(batch_size=16)
val_fds = val_fds.prefetch(buffer_size=AUTOTUNE)

test_fds = test_image_label_ds.shuffle(buffer_size=3750)
#test_fds = test_fds.repeat()
test_fds = test_fds.batch(batch_size=16)
test_fds = test_fds.prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential()
# model.add(tf.keras.layers.Input(shape=(150, 150, 3)))
model.add(tf.keras.layers.Conv2D(input_shape=(150, 150, 3), filters=16, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2))

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(2))

# Flatten feature map to a 1-dim tensor
model.add(layers.Flatten())

# Create a fully connected layer with ReLU activation and 512 hidden units
model.add(layers.Dense(512, activation='relu'))

# Add a dropout rate of 0.5
model.add(layers.Dropout(0.5))

# Create output layer with a single node and sigmoid activation
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.BinaryCrossentropy(),
              metrics=[keras.metrics.binary_accuracy])

steps_per_epoch = tf.math.ceil(len(all_image_paths) / 32).numpy()
print(steps_per_epoch)

history = model.fit(train_fds, epochs=30, validation_data=val_fds,  verbose=2)

# list all data in history
print(history.history.keys())

# Retrieve a list of accuracy results on training and test data
# sets for each training epoch
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']

# Retrieve a list of list results on training and test data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()
plt.show()
# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')
plt.show()

# Evaluate the model on the test data using `evaluate`
print('\n# Evaluate on test data')
results = model.evaluate(test_fds)
print('test loss, test acc:', results)
