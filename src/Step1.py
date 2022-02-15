#!/usr/bin/python3.9

import os, shutil, sys, re, requests, base64, urllib.parse, numpy
from os import path
from modis_tools.auth import ModisSession
from modis_tools.resources import CollectionApi, GranuleApi
from modis_tools.granule_handler import GranuleHandler
from numpy import ndarray
from pandas import DataFrame

def main():
	print("Starting %s..." % sys.argv[0])
	# find and download data
	data_dir = 'data'
	username = input('Earth Data Username: ')
	password = input('Earth Data Password: ')
	for year in range(2015, 2018):
		for month in range(1, 13):
			download_GPM_L3_product('GPM_3IMERGM', '06', year, month, data_dir, username, password)
		download_MODIS_product('MOD21C3', '061', '%s-01-01' % year, '%s-12-31' % year, data_dir, username, password)
		download_MODIS_product('MCD12C1', '006', '%s-01-01' % year, '%s-12-31' % year, data_dir, username, password)
	#
	print("...Done!")

def download_landcover_for_year(year, http_session):
	download_URL = get_landcover_URL_for_year(year)
	#download_URL = 'https://e4ftl01.cr.usgs.gov//MODV6_Cmp_C/MOTA/MCD12C1.006/2018.01.01/MCD12C1.A2018001.006.2019200161458.hdf'
	local_filename = path.join('data', 'landcover-%s.hdf' % year)
	print('Downloading %s to %s...' % (download_URL, local_filename))
	with http_session.get(download_URL, stream=True) as r:
		print('download status code: ', r.status_code)
		r.raise_for_status()
		with open(local_filename, 'wb') as f:
			for chunk in r.iter_content(chunk_size=2**20):
				f.write(chunk)
				print('.')
	print('...Download complete!')


def download_MODIS_product(short_name, version, start_date, end_date, dest_dirpath, username, password):
	os.makedirs(dest_dirpath, exist_ok=True)
	modis_session = ModisSession(username=username, password=password)
	# Query the MODIS catalog for collections
	collection_client = CollectionApi(session=modis_session)
	collections = collection_client.query(short_name=short_name, version=version)
	granule_client = GranuleApi.from_collection(collections[0], session=modis_session)
	granules = granule_client.query(start_date=start_date, end_date=end_date)
	print('Downloading %s to %s...' % (short_name, dest_dirpath))
	GranuleHandler.download_from_granules(granules, modis_session, path='data')
	print('...Download complete!')


def download_GPM_L3_product(short_name, version, year, month, dest_dirpath, username, password):
	# wget --load-cookies /.urs_cookies --save-cookies /root/.urs_cookies --auth-no-challenge=on --user=your_user_name --ask-password --content-disposition -i <url text file>
	http_session = requests.session()
	if(month < 10):
		month_str = '0'+str(month)
	else:
		month_str = str(month)
	src_url = 'https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/%s.%s/%s/3B-MO.MS.MRG.3IMERG.%s%s01-S000000-E235959.%s.V06B.HDF5' % (short_name, version, year, year, month_str, month_str)
	http_session.auth = (username, password)
	# note: URL gets redirected
	redirect = http_session.request('get', src_url)
	filename = src_url.split('/')[-1]
	dest_filepath = path.join(dest_dirpath, filename)
	# NOTE the stream=True parameter below
	print('Downloading %s to %s...' % (redirect.url, dest_filepath))
	with http_session.get(redirect.url, auth=(username, password), stream=True) as r:
		r.raise_for_status()
		with open(dest_filepath, 'wb') as f:
			for chunk in r.iter_content(chunk_size=1048576):
				f.write(chunk)
	print('...Download complete!')



def get_landcover_URL_for_year(year):
	html = requests.get('https://e4ftl01.cr.usgs.gov/MOTA/MCD12C1.006/%s.01.01/' % year).content.decode('UTF-8')
	filename = re.findall('MCD12C1.*?\\.hdf', html)[0]
	return 'https://e4ftl01.cr.usgs.gov/MOTA/MCD12C1.006/%s.01.01/%s' % (year, filename)

if __name__ == '__main__':
	main()