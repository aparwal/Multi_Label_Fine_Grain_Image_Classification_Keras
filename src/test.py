# test.py
# __author__ = Anand Parwal
#

# from keras import applications
# from keras.preprocessing.image import load_img
# from ImageDataGeneratorMultiLabel import ImageDataGeneratorMultiLabel
# from keras import optimizers
from keras.models import Sequential, Model ,load_model
# from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
# from keras import backend as K
import os
from os import environ
import numpy as np
import cv2

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
batch_size = 32
nb_train_samples 	  = 100
nb_validation_samples = 10
epochs = 10
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
# model_base = applications.inception_v3.InceptionV3(weights = "imagenet", include_top=False, input_shape = (img_width, img_height,3), pooling = 'max')

# #Adding custom Layers 
# x = model_base.output
# # x = Flatten()(x)
# # x = Dense(1024, activation="relu")(x)
# # x = Dropout(0.5)(x)
# predictions = Dense(num_classes, activation="sigmoid")(x)

# # first: train only the top layers (which were randomly initialized)
# # i.e. freeze all convolutional InceptionV3 layers
# for layer in model_base.layers:
#     layer.trainable = False

# # this is the model we will train
# model_final = Model(inputs = model_base.input, outputs = predictions)
model_final=load_model('inceptionv3_1.h5')
# model_final.summary()
# compile the model 
# model_final.compile(loss = "binary_crossentropy", optimizer = optimizers.SGD(lr=0.01, momentum=0.9), metrics=['categorical_accuracy'])

print('Model loaded')
no_of_tests = 1
testcases = np.random.randint(100,size = no_of_tests)
imgs = []
for testcase in testcases:
	
	fname = '{name}.jpg'.format(name = testcase)
	img = cv2.imread(os.path.join(TRAIN_DIR, fname)).astype(np.float32)/255.0
	# print(img.shape)
	imgs.append(cv2.resize(img,(img_height,img_width)))
	truth = np.where(y_train[testcase] == 1)
	print(testcase,': ',truth)
imgs = np.array(imgs)
y_prob = model_final.predict(imgs)
y_labels = np.where(y_prob > 0.3)
print(y_labels)
exit()
'''
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
seed = seed,
save_to_dir=TEMP_DIR)

validation_generator = val_datagen.flow_from_directory(
VAL_DIR, VAL_LABEL_PATH,
classes = y_val,
class_mode = 'fromfile',
target_size = (img_height, img_width),
seed = seed)

print('generators initiated')
# Save the model according to the conditions  
checkpoint = ModelCheckpoint("inceptionv3_1.h5", monitor='val_categorical_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0.000, patience=10, verbose=1, mode='auto')


# Train the model 
print("Starting training ...")
model_final.fit_generator(
train_generator,
steps_per_epoch = nb_train_samples,
epochs = epochs,
verbose =1,
validation_data = validation_generator,
validation_steps = nb_validation_samples,
callbacks = [checkpoint, early])
# model_final.fit(x_train, y_train, batch_size=32, epochs=2,callbacks=[checkpoint,early],validation_data=(x_test,y_test))
'''