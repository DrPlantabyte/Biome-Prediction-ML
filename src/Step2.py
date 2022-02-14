#!/usr/bin/python3.9

import os, sys, h5py, numpy, json
from os import path
from matplotlib import pyplot
from osgeo import gdal

def main():
	print("Starting %s..." % sys.argv[0])
	data_dir = path.join('data')
	modis_file = path.join(data_dir, 'MOD21C3.A2015001.061.2021320021656.hdf')
	modis_dataset: gdal.Dataset = gdal.Open(modis_file)
	print_modis_structure(modis_dataset)
	scale_factor = 0.02
	data_name = 'Daytime LST'
	data_map = gdal.Open(modis_dataset.GetSubDatasets()[5][0]).ReadAsArray()
	print(data_map.shape)
	print(data_name, data_map.min(), '-', data_map.max())
	pyplot.clf()
	kelvin = data_map.astype(numpy.float32) * scale_factor
	kelvin[kelvin == 0] = numpy.nan
	kelvin[kelvin > (273.15 + 50)] = numpy.nan
	pyplot.imshow(kelvin - 273.15, origin='upper', cmap='inferno')
	pyplot.colorbar()
	pyplot.title(data_name)
	pyplot.savefig('temperature_map_%s.png' % data_name)
	pyplot.show()

	hdf_file = path.join(data_dir, '3B-MO.MS.MRG.3IMERG.20160801-S000000-E235959.08.V06B.HDF5')
	# note: precipitation data has units of mm/hr, and is the month-long average of per-hour rates
	with h5py.File(hdf_file, 'r') as hdf:
		print_structure(hdf)
		for data_type in ['precipitation', 'randomError', 'gaugeRelativeWeighting', 'probabilityLiquidPrecipitation', 'precipitationQualityIndex']:
			data_map = hdf.get('/Grid/%s' % data_type)[0].T
			# note: -9999.9 means "no data"
			print(data_type, data_map.min(),'-',data_map.max())
			pyplot.clf()
			masked_data = data_map.astype(numpy.float32)
			masked_data[masked_data < 0] = numpy.nan
			pyplot.imshow(masked_data, origin='lower', cmap='gist_rainbow')
			pyplot.colorbar()
			pyplot.title(data_type)
			pyplot.savefig('precip_map_%s.png' % data_type)
			pyplot.show()
	print('...Done!')


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