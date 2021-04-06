# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 23:45:59 2021

@author: brian
"""

import platform

#print(f"Python version: {platform.python_version()}")
assert platform.python_version_tuple() >= ("3", "6")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import random
from random import shuffle
from PIL import Image



# Setup plots
#%matplotlib inline
#plt.rcParams["figure.figsize"] = 10, 8
#%config InlineBackend.figure_format = 'retina'

import tensorflow as tf

#print(f"TensorFlow version: {tf.__version__}")
#print(f"Keras version: {tf.keras.__version__}")

from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Reshape,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
    Dropout
)
from tensorflow.keras.datasets import fashion_mnist

# # Load training inputs from the Fashion-MNIST dataset
(train_images, _), (_, _) = fashion_mnist.load_data()

# # Change pixel values from (0, 255) to (0, 1)
x_train = train_images.astype("float32") / 255

#print(f"x_train: {x_train.shape}")

monet_jpg=[]
photo_jpg=[]
monet_tfrec=[]
photo_tfrec=[]

for file in glob.glob("gan-getting-started/monet_jpg/*.jpg"):
    #print(file)
    monet_jpg.append(file)
    #img = mpimg.imread(file)
    
    #imgplot = plt.imshow(img)
    #plt.show()

for file in glob.glob("gan-getting-started/photo_jpg/*.jpg"):
    photo_jpg.append(file)

#Shuffle the pictures in one list
data_jpg=monet_jpg+photo_jpg
random.shuffle(data_jpg)

for file in glob.glob("gan-getting-started/monet_tfrec/*.tfrec"):
    monet_tfrec.append(file)
    #for example in tf.python_io.tf_record_iterator("data/foobar.tfrecord"):
    #print(tf.train.Example.FromString(file))   
    
for file in glob.glob("gan-getting-started/photo_tfrec/*.tfrec"):
    photo_tfrec.append(file)
    raw_dataset = tf.data.TFRecordDataset(file)

#Display the components of the last raw_dataset     
for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  #print(example)
  
train_images=data_jpg
x_train=[]
for i in range (len(train_images)):
    #img = Image.open(data_jpg[i])[:,:,0]/255
    img=mpimg.imread(data_jpg[i])[:,:,0].astype("float32")/255
    x_train.append(img)

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def train_gan(gan, dataset, batch_size, codings_size, n_epochs=50):
    generator, discriminator = gan.layers
    for epoch in range(n_epochs):
        printProgressBar(epoch+1, n_epochs)
        #print(f"Epoch [{epoch+1}/{n_epochs}]...")
        for x_batch in dataset:
            # Phase 1 - training the discriminator
            noise = tf.random.normal(shape=(batch_size, codings_size))
            generated_images = generator(noise)
            # Gather an equal number of generated (y=0) and real (y=1) images
            x_discr = tf.concat([generated_images, x_batch], axis=0)
            y_discr = tf.constant([[0.0]] * batch_size + [[1.0]] * batch_size)
            # https://stackoverflow.com/a/49100617
            discriminator.train_on_batch(x_discr, y_discr)
            
            # Phase 2 - training the generator
            noise = tf.random.normal(shape=(batch_size, codings_size))
            # Generated images should be labeled "real" by the discriminator
            y_gen = tf.constant([[1.0]] * batch_size)
            # Update only the generator weights (see above)
            gan.train_on_batch(noise, y_gen)
    print("Training complete!")
    
    
codings_size = 100

dcgan_generator = Sequential(
    [
        Dense(7 * 7 * 128, input_shape=(codings_size,)),
        Reshape((7, 7, 128)),
        BatchNormalization(),
        Conv2DTranspose(
            64, kernel_size=5, strides=2, padding="same", activation="selu"
        ),
        BatchNormalization(),
        Conv2DTranspose(1, kernel_size=5, strides=2, padding="same", activation="tanh"),
    ],
    name="generator",
)

dcgan_generator.summary()

dcgan_discriminator = Sequential(
    [
        Conv2D(
            64,
            kernel_size=5,
            strides=2,
            padding="same",
            activation=LeakyReLU(0.2),
            input_shape=(28, 28, 1),
        ),
        Dropout(0.4),
        Conv2D(
            128, kernel_size=5, strides=2, padding="same", activation=LeakyReLU(0.2)
        ),
        Dropout(0.4),
        Flatten(),
        Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)

dcgan_discriminator.summary()

dcgan = Sequential([dcgan_generator, dcgan_discriminator])

dcgan.summary()

# The generator is trained through the GAN model: no need to compile it

dcgan_discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")

# The trainable attribute is taken into account only when compiling a model
# Discriminator weights will be updated only when it will be trained on its own
# They will be frozen when the whole GAN model will be trained
dcgan_discriminator.trainable = False

dcgan.compile(loss="binary_crossentropy", optimizer="rmsprop")

# Reshape and rescale input into a 4D tensor with values between -1 and 1
# Needed because tanh outputs are in this range
x_train_dcgan=[]
for k in range (len(x_train)):
    x_train_dcgan[k] = x_train[k].reshape(-1, 28, 28, 1) * 2. - 1.
    print(f"x_train_dcgan: {x_train_dcgan.shape}")

batch_size = 32

# Load images in batches
dataset = tf.data.Dataset.from_tensor_slices(x_train_dcgan)
dataset = dataset.shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

# Train the DCGAN model
train_gan(dcgan, dataset, batch_size, codings_size, n_epochs=5)

noise = tf.random.normal(shape=(batch_size, codings_size))
generated_images = dcgan_generator(noise)

for k in range (len(generated_images)):
    imgplot = plt.imshow(generated_images[k])
    plt.show()