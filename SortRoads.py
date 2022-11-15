import json
import pandas as pd
from shapely.geometry import Polygon, shape, Point

file = open("./point_lists/Clarke_county.json", 'r')
data = json.load(file)
coords = []

for coord in data:
    coords.append(coord)
file.close()
geometry =  {'type': 'Polygon', 'coordinates': [coords]}
polygon = shape(geometry)

roads_df = pd.read_csv("./road_data/ga.csv", index_col="Unnamed: 0")
county_df = pd.DataFrame(columns=roads_df.columns)
for ind, row in roads_df.iterrows():
    x = row.x
    y = row.y
    point = Point(x,y)
    if (polygon.contains(point)):
        county_df.loc[len(county_df.index)] = [x,y,row.road_id]

print(county_df)
county_df.to_csv("./road_data/Clarke_county_roads.csv")    
    
