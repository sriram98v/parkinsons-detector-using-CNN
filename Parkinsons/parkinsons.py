import numpy as np
import os
import pandas as pd
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Conv1D, Dropout, Flatten
from keras.models import model_from_yaml


try:
	yaml_file = open('parkinsons.yaml', 'r')
	loaded_model_yaml = yaml_file.read()
	yaml_file.close()
	loaded_model = model_from_yaml(loaded_model_yaml)
	loaded_model.load_weights("model.h5")
	print("Loaded model from disk")

except:
	path = os.path.join(os.getcwd()+'/datasets/parkinsons.csv')
	dataframe = pd.read_csv(path)
	
	
	from sklearn.utils import shuffle
	data = shuffle(dataframe, random_state=101)
	x = data.drop(columns=['name', 'status'])
	y = data['status']
	
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)
	
	
	
	batch_size = 1
	num_classes = 2
	epochs = 100
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	
	
	
	X_train = X_train.as_matrix(columns=None).reshape(X_train.shape + (1,))
	X_test = X_test.as_matrix(columns=None).reshape(X_test.shape + (1,))
	
	
	model = Sequential()
	model.add(Conv1D(32, (3), activation='softmax', input_shape=[22, 1]))
	model.add(Conv1D(64, (3), activation='softmax'))
	model.add(Dropout(0.25))
	model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
	model.add(Dense(128, activation='softmax'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	
	
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])
	
	
	model.fit(X_train, y_train,
	          batch_size=batch_size,
	          epochs=epochs,
	          verbose=1,
	          validation_data=(X_test, y_test))
	
	
	scores = model.evaluate(X_test, y_test, verbose=0)
	model_yaml = model.to_yaml()
	with open("parkinsons.yaml", "w") as yaml_file:
	    yaml_file.write(model_yaml)
	# serialize weights to HDF5
	model.save_weights("model.h5")
	print("Saved model to disk")
	