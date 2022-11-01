import csv
import numpy as np
import gmplot
from constants import api_key
import json

# apikey=api_key

# gmap = gmplot.GoogleMapPlotter(33.9511687467, -83.3673290793, 14, apikey=apikey, map_type='hybrid')

file = open('county_data/clarke_county.json', 'r')

data = json.load(file)

fulldata = data['fields']['geo_shape']['coordinates']
# print(fulldata)
print(fulldata[0][0])

coords = [map(float,i.split(',')) for i in fulldata]


file.close()