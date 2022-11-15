import pandas as pd
import geopandas as gpd
import json
from tqdm import tqdm

counties = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip")
print(counties)
for ind, row in counties.iterrows():
    county_name = row.NAME
    county_geo = []
    if row.STATEFP == "13":
        try:
            coords = row.geometry.exterior.coords
            for coord in coords:
                county_geo.append(list(coord))
        except:
            for polygon in row.geometry:
                coords = polygon.exterior.coords
                for coord in coords:
                    county_geo.append(list(coord))
        
        with open('./point_lists/' + county_name + '_' + 'county.json', 'w') as f:
            json.dump(county_geo, f)