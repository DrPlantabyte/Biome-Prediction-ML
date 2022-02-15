#!/usr/bin/python3.9

import os, sys, numpy, pickle
from pandas import DataFrame
from os import path
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
import tensorflow as tf
from tensorflow import keras

def main():
	print("Starting %s..." % sys.argv[0])
	###
	data_dir = path.join('data')
	model_dir = path.join('model')

	data_table: DataFrame = load_pickle(path.join(data_dir, 'data_table.pickle'))
	for col in data_table.columns:
		col_data = data_table[col]
		print('%s\t[%s, %s]' % (col, numpy.min(col_data), numpy.max(col_data)))
	print()
	# now separate the classifier from the rest of the data and normalize it
	normalizer = MinMaxScaler()
	x_data = normalizer.fit_transform(data_table.drop('Classification (IGBP code)', axis=1))
	y_data = numpy.asarray(data_table['Classification (IGBP code)'])
	scaling_vector_slope  = normalizer.data_range_
	scaling_vector_offset = normalizer.data_min_
	print('normalizer vectors [slope offset]:\n', numpy.stack((scaling_vector_slope, scaling_vector_offset), axis=1))
	save_pickle(path.join(model_dir, 'normalizer.pickle'), normalizer)
	print('x_data.shape == %s\ty_data.shape == %s' % (x_data.shape, y_data.shape))

	# now separate training and testing data
	row_count = x_data.shape[0]
	indices = numpy.indices([row_count])[0]
	numpy.random.shuffle(indices)
	x_training = x_data.take(indices[0:int(0.80*row_count)], axis=0)
	y_training = y_data.take(indices[0:int(0.80*row_count)], axis=0)
	x_testing = x_data.take(indices[int(0.80*row_count):row_count], axis=0)
	y_testing = y_data.take(indices[int(0.80*row_count):row_count], axis=0)

	# make the node network model
	model = keras.models.Sequential([
		keras.layers.Dense(300, activation="relu", input_shape=(4,)),
		keras.layers.Dense(100, activation="relu"),
		keras.layers.Dense(17+1, activation="softmax") # +1 because Y data is 1-indexed instead of 0-indexed
	])
	print('input shape:', model.input_shape)
	print('output shape:', model.output_shape)
	model.build()
	print(model.summary())

	model.compile(
		loss=keras.losses.sparse_categorical_crossentropy,
		optimizer=keras.optimizers.SGD(learning_rate=0.01),
		metrics=['accuracy']
	)

	print('Starting to train...')
	print('x_training.shape == %s\ty_training.shape == %s' % (x_training.shape, y_training.shape))
	history = model.fit(x_training, y_training, batch_size=100, epochs=30, validation_split=0.1)
	print('...training done!')
	# see the evolution of the model
	DataFrame(history.history).plot()
	pyplot.grid(True)
	#pyplot.gca().set_ylim(0,1)
	pyplot.xlabel("epoch")
	pyplot.show()

	# measure accuraty with testing data
	test = model.evaluate(x_testing, y_testing) # returns loss, metrics...
	print('Accuracy on test data: %.2f%%' % (100*test[1]))

	model.save(path.join(model_dir, 'biome_model.tf.keras'))

	igbp_names = ['ERROR', 'Evergreen needleleaf forest', 'Evergreen broadleaf forest', 'Deciduous needleleaf forest',
				  'Deciduous broadleaf forest', 'Mixed forest', 'Closed shrubland', 'Open shrubland', 'Woody savanna',
				  'Savanna', 'Grassland', 'Permanent wetland', 'Cropland', 'Urban and built-up landscape',
				  'Cropland/natural vegetation mosaics', 'Snow and ice', 'Barren', 'Water bodies']

	print("Test the prediction model:")
	T_min = float(input("Enter min temperature (C): "))
	T_max = float(input("Enter max temperature (C): "))
	rain = float(input("Enter annual rainfall (mm): "))
	rain_dev = float(input("Enter rainfall std dev (% of average): %"))
	x = normalizer.transform([numpy.asarray([T_min, T_max, rain, rain_dev])])
	class_predictions = model.predict([x])[0]
	print(class_predictions.round(2))
	predicted_biome = numpy.argmax(class_predictions)
	print("Predicted IGBP code: %s (%s)" % (predicted_biome, igbp_names[predicted_biome]))

	rainfalls = [100, 500, 1000]
	rainfall_variations = [10, 25, 50]
	min_temps = numpy.linspace(-20, 50, 71)
	max_temps = numpy.linspace(-20, 50, 71)

	fig, axs = pyplot.subplots(3, 3)
	def predictions(rainfall, rain_var):
		L = len(max_temps)
		df = DataFrame(zip([min_temps, max_temps,[rainfall]*L, [rain_var]*L]), columns=data_table.columns[:-1])
		xd = normalizer.transform(df)
		preds = numpy.argmax(model.predict(xd), axis=0)
		for pt in preds:




	fig.show()





	###
	print('...Done!')


def load_pickle(filepath):
	if path.exists(filepath):
		with open(filepath, 'rb') as fin:
			return pickle.load(fin)
	else:
		return None

def save_pickle(filepath, data):
	dir_path = path.dirname(filepath)
	if path.exists(dir_path) == False:
		os.makedirs(dir_path, exist_ok=True)
	with open(filepath, 'wb') as fout:
		pickle.dump(data, fout)
#
if __name__ == '__main__':
	main()