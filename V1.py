#https://www.kaggle.com/c/gan-getting-started/data
#https://www.bpesquet.fr/mlhandbook/algorithms/generative_adversarial_networks.html


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
from sys import exit


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

# # Load training inputs from the Fashion-MNIST dataset
#(train_images, _), (_, _) = fashion_mnist.load_data()

# # Change pixel values from (0, 255) to (0, 1)
#x_train = train_images.astype("float32") / 255

#print(f"x_train: {x_train.shape}")

###
###----------------LOAD DATA-------------------###
###
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
#data_jpg=monet_jpg+photo_jpg
#random.shuffle(data_jpg)

for k in range (len(monet_jpg)):
    img = Image.open(monet_jpg[k]).convert('RGB')
    img=img.resize((16,16))
    monet_jpg[k]= np.array(img)
for k in range (len(photo_jpg)):
    img=Image.open(photo_jpg[k]).convert('RGB')
    img=img.resize((16,16))
    photo_jpg[k]= np.array(img)
    


for file in glob.glob("gan-getting-started/monet_tfrec/*.tfrec"):
    monet_tfrec.append(file)
    #for example in tf.python_io.tf_record_iterator("data/foobar.tfrecord"):
    #print(tf.train.Example.FromString(file))  
#    raw_dataset_monet = tf.data.TFRecordDataset(file)
    
for file in glob.glob("gan-getting-started/photo_tfrec/*.tfrec"):
    photo_tfrec.append(file)
    raw_dataset = tf.data.TFRecordDataset(file)
#    raw_dataset_photo = tf.data.TFRecordDataset(file)

###
###
###---------------------------------------------###

###
###---------------PREPARING THE DATA------------###
###

train_images=[]
train_labels=[]
test_images=[]
test_labels=[]

for k in range (59): #20% of the database
    test_images.append(monet_jpg[k])
for i in range (59,300):
    train_images.append(monet_jpg[i])
for k in range (1409):
    test_images.append(photo_jpg[k])
for i in range(1409,7038):
    train_images.append(photo_jpg[k])



#Display the components of the last raw_dataset     
for raw_record in raw_dataset.take(1):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  #print(example)
  

x_train=[]
for i in range (len(train_images)):
    #img = Image.open(data_jpg[i])[:,:,0]/255
    img=train_images[i][:,:,0].astype("float32")/255
    #img=mpimg.imread(train_images[i])[:,:,0].astype("float32")/255
    x_train.append(img)

x_test=[]
for i in range (len(test_images)):
    #img = Image.open(data_jpg[i])[:,:,0]/255
    img=test_images[i][:,:,0].astype("float32")/255
    #img=mpimg.imread(test_images[i])[:,:,0].astype("float32")/255
    x_test.append(img)
    
print(f'x_train: ({len(x_train)},{x_train[0].shape}). x_test: ({len(x_test)},{x_test[0].shape})')

    
codings_size = 30
image_shape=16

###
###------------------DEFINE A MODEL----------------------###
###
# generator = Sequential()
# model.add(Flatten(input_shape=(16, 16)))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(10, activation='softmax'))

# model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


generator = Sequential(
    [
        Dense(100, activation="selu", input_shape=(codings_size,)),
        #Dense(100, activation="selu", input_shape=(image_shape, image_shape)),
        Dense(150, activation="selu"),       
        Dense(image_shape * image_shape, activation="sigmoid"),
        #Dense(image_shape, activation="sigmoid"),
        Reshape((image_shape, image_shape)),
    ],
    name="generator"
)

discriminator = Sequential(
    [
        Flatten(input_shape=(image_shape, image_shape)),
        Dense(150, activation="selu"),
        Dense(100, activation="selu"),
        Dense(1, activation="sigmoid"),
    ],
    name="discriminator"
)

gan = Sequential([generator, discriminator])

# Print GAN model summary
gan.summary()

# The generator is trained through the GAN model: no need to compile it
discriminator.compile(loss="binary_crossentropy", optimizer="rmsprop")

# The trainable attribute is taken into account only when compiling a model
# Discriminator weights will be updated only when it will be trained on its own
# They will be frozen when the whole GAN model will be trained
discriminator.trainable = False

gan.compile(loss="binary_crossentropy", optimizer="rmsprop")



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
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
        print()
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

    print("Entrainement terminé !")
    
batch_size = 32


# Load images in batches
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000)
dataset = dataset.batch(batch_size, drop_remainder=True).prefetch(1)

#gan.evaluate(x=dataset, batch_size=batch_size)
# Train the GAN model
train_gan(gan, dataset, batch_size, codings_size, n_epochs=50)
#history=gan.fit(x = x_train, epochs = 50) #add batch_size then
#print(history.history.keys())

noise = tf.random.normal(shape=(batch_size, codings_size))
#noise = tf.random.normal(shape=(image_shape, image_shape))
generated_images = generator(noise)

tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0,
    patience=0,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

for k in range (len(generated_images)):
    #img=generated_images[k][:,:,0].astype("float32")*255
    imgplot = plt.imshow(generated_images[k],cmap=plt.cm.binary)    
    plt.show()
    




