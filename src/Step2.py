#!/usr/bin/python3.9

import os, sys, h5py
from os import path

def main():
	print("Starting %s..." % sys.argv[0])
	data_dir = path.join('data')
	hdf_file = path.join(data_dir, '3B-MO.MS.MRG.3IMERG.20160801-S000000-E235959.08.V06B.HDF5')
	with h5py.File(hdf_file, 'r') as hdf:
		print_structure(hdf)
	print('...Done!')


def _print_structure(node, indent):
	print('\t'*indent, end='')
	print(node.name)
	for key in node.keys():
		_print_structure(node.get(key), indent+1)
def print_structure(hdf: h5py.File):
	for key in hdf.keys():
		_print_structure(hdf.get(key), 0)

if __name__ == '__main__':
	main()