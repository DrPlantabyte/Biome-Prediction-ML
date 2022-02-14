#!/usr/bin/python3.9

import os, sys, h5py, numpy
from os import path
from matplotlib import pyplot

def main():
	print("Starting %s..." % sys.argv[0])
	data_dir = path.join('data')
	hdf_file = path.join(data_dir, '3B-MO.MS.MRG.3IMERG.20160801-S000000-E235959.08.V06B.HDF5')
	# note: precipitation data has units of mm/hr, and is the month-long average of per-hour rates
	with h5py.File(hdf_file, 'r') as hdf:
		print_structure(hdf)
		for data_type in ['precipitation', 'randomError', 'gaugeRelativeWeighting', 'probabilityLiquidPrecipitation', 'precipitationQualityIndex']:
			data_map = hdf.get('/Grid/%s' % data_type)[0].T
			# note: -9999.9 means "no data"
			print(data_type, data_map.min(),'-',data_map.max())
			pyplot.clf()
			pyplot.imshow(numpy.square(numpy.sqrt(data_map)), origin='lower', cmap='gist_rainbow')
			pyplot.colorbar()
			pyplot.title(data_type)
			pyplot.savefig('precip_map_%s.png' % data_type)
			pyplot.show()
	print('...Done!')


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