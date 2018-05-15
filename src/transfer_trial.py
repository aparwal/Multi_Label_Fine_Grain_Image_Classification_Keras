#!/bin/env python3
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -p gpu
#SBATCH -o train-deeper150.out
#SBATCH -t 22:20:00
#SBATCH --gres=gpu:2
#SBATCH --mem=106000

#####################
# train.py
# transfer learning for multilabel classification
# __author__ = Anand Parwal
#

#for setting directory on slurm/cluster
import sys
import os
sys.path.append(os.getcwd())

from keras import applications
# from keras.preprocessing.image import ImageDataGenerator
from ImageDataGeneratorMultiLabel import ImageDataGeneratorMultiLabel
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from os import environ
import numpy as np


# K.set_image_dim_ordering('th')

# Specify directories
DATA_DIR  = '../data/'
TRAIN_DIR = DATA_DIR + 'train/'
VAL_DIR   = DATA_DIR + 'validation/'
TEST_DIR  = DATA_DIR + 'test/'
TEMP_DIR  = DATA_DIR + 'tempdata/'
TRAIN_LABEL_PATH = DATA_DIR + 'label_train.npy'
VAL_LABEL_PATH 	 = DATA_DIR + 'label_val.npy'

# Specify network parameters
img_width, img_height = 299, 299

# Specify training parameters
batch_size = 64
nb_train_samples 	  = 156#20
nb_validation_samples = 100
epochs = 150
seed   = 7

# reduce level of log from tensorflow
environ['TF_CPP_MIN_LOG_LEVEL']='1'

##################################################
# Load label data
y_train = np.load(TRAIN_LABEL_PATH)
y_val = np.load(VAL_LABEL_PATH)
# Number of classes is the length of the label of any image
num_classes = len(y_train[0])
print('Ground truth loaded')
###################################################

# Load the pretrained inceptionv3 network without the fully connected layers on top
model_base = applications.inception_v3.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height,3), pooling = 'max')

#Adding custom Layers 
x = model_base.output
# x = Flatten()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation="sigmoid")(x)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in model_base.layers:
    layer.trainable = False

# this is the model we will train
model_final = Model(inputs = model_base.input, outputs = predictions)
# model_final.summary()
# compile the model 
model_final.compile(loss = "binary_crossentropy", optimizer = 'Nadam', metrics=['categorical_accuracy'])

print('Model compiled')
# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGeneratorMultiLabel(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

val_datagen = ImageDataGeneratorMultiLabel(
rescale = 1./255)
# horizontal_flip = True,
# fill_mode = "nearest",
# zoom_range = 0.3,
# width_shift_range = 0.3,
# height_shift_range=0.3,
# rotation_range=30)

train_generator = train_datagen.flow_from_directory(
TRAIN_DIR, TRAIN_LABEL_PATH,
classes = y_train,
class_mode = 'fromfile',
target_size = (img_height, img_width),
batch_size = batch_size, 
seed = seed)

validation_generator = val_datagen.flow_from_directory(
VAL_DIR, VAL_LABEL_PATH,
classes = y_val,
class_mode = 'fromfile',
target_size = (img_height, img_width),
seed = seed)

print('generators initiated')
# Save the model according to the conditions  
checkpoint = ModelCheckpoint("inceptionv3_21.{epoch:02d}-{loss:.2f}.h5", monitor='val_categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='loss', min_delta=0.000, patience=10, verbose=1, mode='auto')
tensorbd  = TensorBoard(log_dir="logs/train/{}-layers_{}-batch_{}-samples ".format(len(model_final.layers),batch_size,nb_train_samples),
						batch_size=batch_size)

# Train the model 
print("Starting training ...")
model_final.fit_generator(
train_generator,
steps_per_epoch = nb_train_samples,
epochs = epochs,
verbose =1,
validation_data = validation_generator,
validation_steps = nb_validation_samples,
callbacks = [checkpoint, early, tensorbd])
# model_final.fit(x_train, y_train, batch_size=32, epochs=2,callbacks=[checkpoint,early],validation_data=(x_test,y_test))
