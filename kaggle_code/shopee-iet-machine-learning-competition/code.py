import csv
from keras.preprocessing.image import img_to_array
from keras import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from imutils import paths
from keras import Model
from keras import applications
import numpy as np
import argparse
import imutils
import vgg
import cv2

img_width, img_height = 224, 224
INIT_LR = 1e-4

top_model_weights_path = 'bottleneck_fc_model1.h5'
train_data_dir = 'data/Training Images'
validation_data_dir = 'data/Training Images copy'
# nb_train_samples = 33894
# nb_validation_samples = 4317
nb_train_samples = 33888
nb_validation_samples = 4288
epochs = 30
batch_size = 32


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')
    print(model.summary())
    print("loaded network")
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print("")
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    print(type(bottleneck_features_train))
    np.save('bottleneck_features_train.npy',
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save('bottleneck_features_validation.npy',
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load('bottleneck_features_train.npy')
    # train_labels = np.array(
    #     [0] * 3099 + [1] * 2399 + [2] * 1699 + [3] * 1699 + [4] * 1699 + [5] * 1900 + [6] * 2900 + [7] * 2900 + [8] * 1199 + [9] * 2000 +  [10] * 2700 + [11] * 2700 + [12] * 1600 + [13] * 1400 + [14] * 1400 + [15] * 1000 + [16] * 700 + [17] * 900)
    train_labels = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 3099 + [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 2399 + [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 1699 + [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 1699
                            + [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 1699 + [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 1900 + [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 2900 + [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 2900
                            + [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 1199 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]] * 2000 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]] * 2700 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]] * 2700
                            + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]] * 1600 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]] * 1400 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]] * 1400 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]] * 1000
                            + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] * 700 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]] * 894)
    validation_data = np.load('bottleneck_features_validation.npy')
    # validation_labels = np.array(
    #     [0] * 289 + [1] * 250 + [2] * 227 + [3] * 213 + [4] * 275 + [5] * 212 + [6] * 201 + [7] * 288 + [
    #         8] * 248 + [9] * 289 + [10] * 288 + [11] * 216 + [12] * 220 + [13] * 218 + [14] * 280 + [
    #         15] * 294 + [16] * 110 + [17] * 199)

    validation_labels = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 289 + [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 250 + [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 227 + [[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 213
                            + [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 275 + [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 212 + [[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 201 + [[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 288
                            + [[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]] * 248 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]] * 289 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]] * 288 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]] * 216
                            + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]] * 220 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]] * 218 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]] * 280 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]] * 294
                            + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]] * 110 + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]] * 170)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(2048, activation='relu'))
    model.add(Dropout(0.8))
    # model.add(Dense(512, activation = "relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(18, activation='softmax'))


    opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])
    print(model.summary())
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


train_top_model()

weights_path = 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
top_model_weights_path = 'bottleneck_fc_model1.h5'
# dimensions of our images.
img_width, img_height = 224, 224

train_data_dir = 'data/Training Images'
validation_data_dir = 'data/Training Images copy'
nb_train_samples = 33894
nb_validation_samples = 4317
epochs = 50
batch_size = 32

# build the VGG16 network
model = vgg.VGG_16()
model.load_weights("vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
print('Model loaded.')

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=(7, 7, 512)))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dropout(0.6))
top_model.add(Dense(4096, activation='relu'))
top_model.add(Dropout(0.7))
top_model.add(Dense(18, activation='softmax'))
top_model.load_weights(top_model_weights_path)

model.add(top_model)

# set the first 25 layers (up to the last conv block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:11]:
    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    rotation_range=10,
    zoom_range=0.2,
    height_shift_range=0.1,
    width_shift_range=0.1,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# fine-tune the model
model.fit_generator(
    train_generator,
    samples_per_epoch=nb_train_samples,
    epochs=epochs,
    validation_data=validation_generator,
    nb_val_samples=nb_validation_samples)

model.save("vgg2.model")

lists = ["BabyBibs", "BabyHat", "BabyPants", "BabyShirt", "PackageFart", "womanshirtsleeve", "womencasualshoes", "womenchiffontop", "womendollshoes", "womenknittedtop", "womenlazyshoes", "womenlongsleevetop", "womenpeashoes", "womenplussizedtop", "womenpointedflatshoes", "womensleevelesstop", "womenstripedtop", "wrapsnslings"]

print("[INFO] loading network...")

model = load_model("vgg2.model")

answerlist = [0 for i in range(16111)]
imagepaths = list(paths.list_images('Test'))
# pre-process the image for classification
count = 0
for imagepath in imagepaths:
	print("{} images done".format(count))
	count = count + 1
	number = int(imagepath.split('.')[0][10:])
	image = cv2.imread(imagepath)
	image = cv2.resize(image, (299, 299))
	image = image.astype("float") / 255.0
	image = img_to_array(image)
	image = np.expand_dims(image, axis=0)
	problist = model.predict(image)[0]
	max = -1
	maxIndex = 0
	for i in range(18):
		if problist[i] > max:
			max = problist[i]
			maxIndex = i
	answerlist[number - 1] = maxIndex


printlist = [[str(i + 1), str(answerlist[i])] for i in range(16111)]

with open('result19.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(["id", "category"])
    spamwriter.writerows(printlist)