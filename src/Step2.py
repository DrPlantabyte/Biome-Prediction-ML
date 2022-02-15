#!/usr/bin/python3.9

import os, sys, h5py, numpy, json, math, pickle
from pandas import DataFrame
from os import path
from matplotlib import pyplot
from osgeo import gdal

def main():
	print("Starting %s..." % sys.argv[0])
	data_dir = path.join('data')
	'''
	biome_map_file = path.join(data_dir, 'MCD12C1.A2015001.006.2018053185652.hdf')
	biome_map_ds = gdal.Open(biome_map_file)
	print_modis_structure(biome_map_ds)
	biome_map = get_modis_data(biome_map_ds, 0)
	plot_data_map(biome_map, 'IGBP cover type', origin='upper', cmap='gist_rainbow')
	biome_map_ds = None # GDAL implements .Close() on object de-reference
	del biome_map_file
	del biome_map
	del biome_map_ds

	sample_LST_file = path.join(data_dir, 'MOD21C3.A2016061.061.2021346202936.hdf')
	LST_ds = gdal.Open(sample_LST_file)
	print_modis_structure(LST_ds)
	LST_map = get_modis_data(LST_ds, 5).astype(numpy.float32) * 0.02
	LST_map[LST_map <= 0] = numpy.nan
	plot_data_map(LST_map-273.15, 'daytime land surface temperature', origin='upper', cmap='inferno')
	LST_ds = None # GDAL implements .Close() on object de-reference
	del sample_LST_file
	del LST_map
	del LST_ds


	sample_rainfall_file = path.join(data_dir, '3B-MO.MS.MRG.3IMERG.20160801-S000000-E235959.08.V06B.HDF5')
	# note: precipitation data has units of mm/hr, and is the month-long average of per-hour rates
	with h5py.File(sample_rainfall_file, 'r') as hdf:
		print_structure(hdf)
		# ['precipitation', 'randomError', 'gaugeRelativeWeighting', 'probabilityLiquidPrecipitation', 'precipitationQualityIndex']
		data_type = 'precipitation'
		data_map = hdf.get('/Grid/%s' % data_type)[0].T
		# note: -9999.9 means "no data"
		print(data_type, data_map.min(),'-',data_map.max())
		masked_data = data_map.astype(numpy.float32)
		masked_data[masked_data < 0] = numpy.nan
		plot_data_map(masked_data * (24*30), '30-day Precipitation', origin='lower', cmap='gist_rainbow')
	'''

	## cover type dimensions: [3600x7200]
	## LST dimensions: [3600x7200]
	## precipitation dimensions: (1, 3600, 1800)

	# calculate min and max temperatures
	min_temp_map = load_pickle(path.join(data_dir, 'min_temp_map.pickle'))
	max_temp_map = load_pickle(path.join(data_dir, 'max_temp_map.pickle'))
	if min_temp_map is None or max_temp_map is None:
		min_temp_map = numpy.zeros((3600,7200), dtype=numpy.float32)
		max_temp_map = numpy.zeros_like(min_temp_map)
		lst_files = [x for x in os.listdir(data_dir) if x.startswith('MOD21C3')]
		lst_date_dict = {}
		for f in lst_files:
			ds: gdal.Dataset = gdal.Open(path.join(data_dir, f))
			ddate = ds.GetMetadata_Dict()["RANGEBEGINNINGDATE"]
			yearmo = ddate[0:4]+ddate[5:7]
			print(yearmo)
			lst_date_dict[yearmo] = f
			ddate = None
			ds = None
		count = 0
		for year in range(2015, 2018):
			print('processing year %s temperature...' % year)
			count += 1
			annual_max_temp_map = None
			annual_min_temp_map = None
			for month in range(1, 13):
				print('\tMonth %s...' % month)
				yearmo = str(year) + to2digit(month)
				lst_filename = lst_date_dict[yearmo]
				ds: gdal.Dataset = gdal.Open(path.join(data_dir, lst_filename))
				daytime_lst = get_modis_data(ds, 5).astype(numpy.float32) * 0.02
				daytime_lst[daytime_lst < 150] = numpy.nan # remove bad values
				nighttime_lst = get_modis_data(ds, 6).astype(numpy.float32) * 0.02
				nighttime_lst[nighttime_lst < 150] = numpy.nan
				if annual_max_temp_map is None:
					annual_max_temp_map = daytime_lst
				if annual_min_temp_map is None:
					annual_min_temp_map = nighttime_lst
				annual_max_temp_map = numpy.nanmax((annual_max_temp_map, daytime_lst, nighttime_lst), axis=0)
				annual_min_temp_map = numpy.nanmin((annual_min_temp_map, daytime_lst, nighttime_lst), axis=0)
				del daytime_lst
				del nighttime_lst
				ds = None
				del ds
			min_temp_map = min_temp_map + annual_min_temp_map
			max_temp_map = max_temp_map + annual_max_temp_map
		min_temp_map = min_temp_map / count
		max_temp_map = max_temp_map / count
		save_pickle(path.join(data_dir, 'min_temp_map.pickle'), min_temp_map)
		save_pickle(path.join(data_dir, 'max_temp_map.pickle'), max_temp_map)
	plot_data_map(min_temp_map-273.15, 'Min temperature', origin='upper', cmap='jet')
	plot_data_map(max_temp_map-273.15, 'Max temperature', origin='upper', cmap='jet')

	# calculate average and std dev of rainfall
	ave_precip_map = load_pickle(path.join(data_dir, 'ave_precip_map.pickle'))
	stdev_precip_map = load_pickle(path.join(data_dir, 'stdev_precip_map.pickle'))
	if ave_precip_map is None or stdev_precip_map is None:
		precip_time_series = None
		for year in range(2015, 2018):
			print('processing year %s rainfall...' % year)
			for month in range(1, 13):
				print('\tMonth %s...' % month)
				precip_filename = '3B-MO.MS.MRG.3IMERG.%s%s01-S000000-E235959.%s.V06B.HDF5' % (year, to2digit(month), to2digit(month))
				with h5py.File(path.join(data_dir, precip_filename), 'r') as hdf:
					## correct to same orientation as modis data
					precip_map = numpy.flip(hdf.get('/Grid/precipitation')[0].T, axis=0).astype(numpy.float32)
					#precip_map = numpy.flip(hdf.get('/Grid/gaugeRelativeWeighting')[0].T, axis=0).astype(numpy.float32)
					precip_map[precip_map < 0] = numpy.nan
					precip_map = precip_map * (24 * 365.24/12) # convert to monthly total
					if precip_time_series is None:
						precip_time_series = precip_map
					elif len(precip_time_series.shape) == 2:
						#
						precip_time_series = numpy.stack((precip_time_series, precip_map), axis=0)
					else:
						precip_time_series = numpy.concatenate((precip_time_series, [precip_map]))
		ave_precip_map = numpy.mean(precip_time_series, axis=0)
		stdev_precip_map = numpy.std(precip_time_series, axis=0)
		save_pickle(path.join(data_dir, 'ave_precip_map.pickle'), ave_precip_map)
		save_pickle(path.join(data_dir, 'stdev_precip_map.pickle'), stdev_precip_map)
	plot_data_map(ave_precip_map, 'Ave rainfall', origin='upper')
	plot_data_map(stdev_precip_map, 'Rainfall std dev', origin='upper')

	# retrieve biomes from 2017
	biome_map_file = path.join(data_dir, 'MCD12C1.A2017001.006.2019192025407.hdf')
	biome_map_ds = gdal.Open(biome_map_file)
	biome_map = numpy.copy(get_modis_data(biome_map_ds, 0))+1
	print('found biome codes: %s' % numpy.unique(biome_map))
	biome_map_ds = None
	del biome_map_ds

	plot_data_map(biome_map, 'classification', origin='upper', cmap='jet')


	# sample with sinusoidal projection
	min_temps = numpy.asarray([], dtype=numpy.float32)
	max_temps = numpy.asarray([], dtype=numpy.float32)
	ave_rain = numpy.asarray([], dtype=numpy.float32)
	dev_rain = numpy.asarray([], dtype=numpy.float32)
	biomes = numpy.asarray([], dtype=numpy.uint8)
	deg2rad = math.pi/180
	rad2deg = 180/math.pi
	spatial_resolution_degrees = 0.1
	for lat in numpy.linspace(-90,90-spatial_resolution_degrees,int(180/spatial_resolution_degrees)):
		# note: Y = 0 is north pole, positive latitude is northern hemisphere
		# note: rain map is 0.1 degree pixels, others are 0.05 degree pixels
		longitudes = numpy.linspace(-180, 180-spatial_resolution_degrees, int(rad2deg*numpy.cos(lat*deg2rad)))
		if len(longitudes) == 0: # don't sample the poles
			continue
		#print('latitude %s (%s longitudes sampled)' % (lat, len(longitudes)))
		biome_row = int((90 - lat) * 20)
		lst_row = int((90 - lat) * 20)
		precip_row = int((90 - lat) * 10)
		biome_cols = ((longitudes + 180) * 20).astype(dtype=numpy.int32)
		lst_cols = ((longitudes + 180) * 20).astype(dtype=numpy.int32)
		precip_cols = ((longitudes + 180) * 10).astype(dtype=numpy.int32)
		min_temps = numpy.concatenate((min_temps, min_temp_map[lst_row].take(lst_cols)))
		max_temps = numpy.concatenate((max_temps, max_temp_map[lst_row].take(lst_cols)))
		ave_rain = numpy.concatenate((ave_rain, ave_precip_map[precip_row].take(precip_cols)))
		dev_rain = numpy.concatenate((dev_rain, stdev_precip_map[precip_row].take(precip_cols)))
		biomes = numpy.concatenate((biomes, biome_map[biome_row].take(biome_cols)))
	data_table: DataFrame = DataFrame.from_dict({
		"Temperature Min (C)": min_temps-273.15,
		"Temperature Max (C)": max_temps-273.15,
		"Annual Rainfall (mm/yr)": ave_rain*12,
		"Monthly Rainfall Std. Dev. (% mean)": (dev_rain/ave_rain) * 100,
		"Classification (IGBP code)": biomes
	})
	## now remove rows with nans
	data_table = data_table.dropna()
	save_pickle(path.join(data_dir, 'data_table.pickle'), data_table)
	print(data_table)



	print('...Done!')

def load_pickle(filepath):
	if path.exists(filepath):
		with open(filepath, 'rb') as fin:
			return pickle.load(fin)
	else:
		return None

def save_pickle(filepath, data):
	with open(filepath, 'wb') as fout:
		pickle.dump(data, fout)


def to2digit(n):
	if (n < 10):
		return '0' + str(n)
	else:
		return str(n)

def plot_data_map(data_map: numpy.ndarray, title: str, origin='lower', cmap='gist_rainbow'):
	pyplot.clf()
	pyplot.imshow(data_map, origin=origin, cmap=cmap)
	pyplot.colorbar()
	pyplot.title(title)
	pyplot.savefig('%s.png' % title)
	pyplot.show()

def get_modis_data(modis_root_dataset: gdal.Dataset, subset_index):
	data_map = gdal.Open(modis_root_dataset.GetSubDatasets()[subset_index][0]).ReadAsArray()
	return data_map

def print_modis_structure(dataset: gdal.Dataset):
	metadata_dict = dict(dataset.GetMetadata_Dict())
	metadata_dict['Subsets'] = dataset.GetSubDatasets()
	print(dataset.GetDescription(), json.dumps(metadata_dict, indent="  "))


def print_structure(hdf: h5py.File):
	for key in hdf.keys():
		_print_structure(hdf.get(key), 0)


def _print_structure(node, indent):
	# recursive implementation
	print('\t'*indent, end='')
	print(node.name)
	if type(node) == h5py._hl.group.Group:
		for key in node.keys():
			_print_structure(node.get(key), indent + 1)
	elif type(node) == h5py._hl.dataset.Dataset:
		print('\t' * (indent+1), end='')
		print('Dataset: dtype = %s; shape = %s; compression = %s; total bytes = %s' % (node.dtype, node.shape, node.compression, node.nbytes))
	else:
		# unknown type
		print('\t' * (indent+1), end='')
		print(type(node))
	pass

if __name__ == '__main__':
	main()