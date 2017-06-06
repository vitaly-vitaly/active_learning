from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
np.random.seed(2017) 
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

# def full_model(X_tr, y_tr, X_tst, y_tst):
# 	X_train = X_tr.reshape((X_tr.shape[0], 3, 32, 32))
# 	X_test = X_tst.reshape((X_tst.shape[0], 3, 32, 32))
# 	y_train = np_utils.to_categorical(y_tr)
# 	y_test = np_utils.to_categorical(y_tst)
# 	num_classes = y_test.shape[1]
# 	# Create the model
# 	model = Sequential()
# 	model.add(Conv2D(32, 3, 3, input_shape=(3, 32, 32), activation='relu'))
# 	model.add(Dropout(0.2))
# 	model.add(Conv2D(32, 3, 3, activation='relu'))
# 	model.add(MaxPooling2D(pool_size=(2, 2)))
# 	model.add(Flatten())
# 	model.add(Dense(512, activation='relu'))
# 	model.add(Dropout(0.5))
# 	model.add(Dense(num_classes, activation='softmax'))
# 	# Compile model
# 	epochs = 20
# 	lrate = 0.01
# 	decay = lrate/epochs
# 	sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
# 	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
# 	# Fit the model
# 	model.fit(X_train, y_train, nb_epoch=epochs, batch_size=32)
# 	# Final evaluation of the model
# 	scores = model.evaluate(X_test, y_test, verbose=0)
# 	# print("Accuracy: %.2f%%" % (scores[1]*100))
# 	return(scores[1])

def full_model(X_tr, y_tr, X_tst, y_tst):
	X_train = X_tr.reshape((X_tr.shape[0], 1, 28, 28))
	X_test = X_tst.reshape((X_tst.shape[0], 1, 28, 28))
	y_train = np_utils.to_categorical(y_tr)
	y_test = np_utils.to_categorical(y_tst)
	num_classes = y_test.shape[1]
	# Create the model
	model = Sequential()
	model.add(Convolution2D(48, 3, 3, border_mode='same', input_shape=(1, 28, 28)))
	model.add(Activation('relu'))
	model.add(Convolution2D(48, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Convolution2D(96, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(96, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Convolution2D(192, 3, 3, border_mode='same'))
	model.add(Activation('relu'))
	model.add(Convolution2D(192, 3, 3))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(512))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(256))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile the model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	# Compile model
	epochs = 20
	# lrate = 0.01
	# decay = lrate/epochs
	# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	# Fit the model
	model.fit(X_train, y_train, nb_epoch=epochs, batch_size=128)
	# Final evaluation of the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	# print("Accuracy: %.2f%%" % (scores[1]*100))
	return(scores[1])












