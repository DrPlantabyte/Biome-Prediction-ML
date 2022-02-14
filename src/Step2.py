#!/usr/bin/python3.9

import os, sys, h5py, numpy, json, math
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
	min_temp_map = numpy.zeros((3600,7200), dtype=numpy.float32)
	max_temp_map = numpy.zeros_like(min_temp_map)


	# sample with sinusoidal projection
	## create coordinate list
	deg2rad = math.pi/180
	rad2deg = 180/math.pi
	spatial_resolution_degrees = 0.1
	coords = numpy.asarray([[0.0,0.0]])
	for lat in numpy.linspace(-90,90,int(180/spatial_resolution_degrees)):
		longitudes = numpy.linspace(-180, 180, int(rad2deg*numpy.cos(lat*deg2rad)))
		latitudes  = numpy.ones_like(longitudes) * lat
		coords = numpy.concatenate((coords, numpy.stack((longitudes, latitudes), axis=1)))
	print(coords)
	print(coords.shape)


	print('...Done!')

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