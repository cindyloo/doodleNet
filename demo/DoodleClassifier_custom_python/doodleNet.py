f = open("categories.txt","r")
# And for reading use
classes = f.readlines()
f.close()

classes = [c.replace('\n','').replace(' ','_') for c in classes]
print(classes)


import requests
import shutil
import os


def download():
    base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
    for cl in classes:
        c = cl.strip('\n')
        cls_url = cl.replace('_', '%20')
        path = base + cls_url + '.npy'

        file_name = 'data/{}.npy'.format(c)
        print(file_name)
        if not os.path.isfile(file_name):
            print(path)
            data = requests.get(path)
            open(file_name, 'wb').write(data.content)


#download()

import os
import glob
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf


class_names = []

def load_data(root, vfold_ratio=0.1, max_items_per_class=5):
    all_files = glob.glob(os.path.join(root, '*.npy'))

    # initialize variables
    x = np.empty([0, 784])
    y = np.empty([0])

    # load each data file
    for idx, file in enumerate(all_files):
        print(idx)
        print(file)
        try:
            data = np.load(file, allow_pickle=True)
            data = data[0: max_items_per_class, :]
            labels = np.full(data.shape[0], idx)

            x = np.concatenate((x, data), axis=0)
            y = np.append(y, labels)

            class_name, ext = os.path.splitext(os.path.basename(file))
            class_names.append(class_name)
        except:
            print("error, continuing")

    data = None
    labels = None

    # randomize the dataset
    permutation = np.random.permutation(y.shape[0])
    x = x[permutation, :]
    y = y[permutation]

    # separate into training and testing
    vfold_size = int(x.shape[0] / 100 * (vfold_ratio * 100))

    x_test = x[0:vfold_size, :]
    y_test = y[0:vfold_size]

    x_train = x[vfold_size:x.shape[0], :]
    y_train = y[vfold_size:y.shape[0]]
    return x_train, y_train, x_test, y_test, class_names

x_train, y_train, x_test, y_test, class_names = load_data('data')
num_classes = len(class_names) #important to get right for Input shapes!
CLASS_SIZE = num_classes
print(num_classes)
image_size = 28

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))


import matplotlib.pyplot as plt
from random import randint
idx = randint(0, len(x_train))
print(x_train[idx].reshape(28,28))
plt.imshow(x_train[idx].reshape(28,28))
print(class_names[int(y_train[idx].item())])


# Reshape and normalize
x_train = x_train.reshape(x_train.shape[0], image_size, image_size, 1).astype('float32')
x_test = x_test.reshape(x_test.shape[0], image_size, image_size, 1).astype('float32')

x_train /= 255.0
x_test /= 255.0

# Convert class vectors to class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential([
    layers.Convolution2D(16, (3, 3),padding='same',input_shape=x_train.shape[1:], activation='relu'),
    layers.Convolution2D(16, (3, 3),padding='same',input_shape=x_train.shape[1:], activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Convolution2D(32, (3, 3), padding='same', activation= 'relu'),
    layers.Convolution2D(32, (3, 3), padding='same', activation= 'relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Convolution2D(64, (3, 3), padding='same', activation= 'relu'),
    layers.Convolution2D(64, (3, 3), padding='same', activation= 'relu'),
    layers.MaxPooling2D(pool_size =(2,2)),

    layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(512, activation='tanh'),
    layers.Dense(CLASS_SIZE, activation='softmax')
    #layers.Softmax(input_shape=CLASS_SIZE)
])

# Train model
adam = tf.compat.v1.train.AdamOptimizer()
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['top_k_categorical_accuracy'])
print(model.summary())

# we have to run the fit method several times to get good predictions
# I don't know if it is better to run this entire module (doodleTrain.py) over several times or
# whether I should parse up the training and testing images and run the fit method
# several times *within* this module

model.fit(x = x_train, y = y_train, validation_split=0.1, batch_size = 256, verbose=2, epochs=10)

print(len(x_train))
print(len(y_train))
print(len(x_test))
print(len(y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test accuarcy: {:0.2f}%'.format(score[1] * 100))


import matplotlib.pyplot as plt
from random import randint
#%matplotlib inline
idx = randint(0, len(x_test))
img = x_test[idx]
plt.imshow(img.squeeze())
pred = model.predict(np.expand_dims(img, axis=0))[0]
ind = (-pred).argsort()[:5]
latex = [class_names[x] for x in ind]
print(latex)


with open('class_names.txt', 'w') as file_handler:
    for item in class_names:
        file_handler.write("{}\n".format(item))


model.save('keras.h5')
#!mkdir model
#!tensorflowjs_converter --input_format keras keras.h5 model/
#!cp class_names.txt model/class_names.txt

"""
export CUDNN_PATH=/usr/local/cuda/lib64
export CUDNN_INCLUDE_DIR=/usr/local/cuda/include
export CUDNN_LIBRARY=/usr/local/cuda/lib64
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
"""