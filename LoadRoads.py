import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

points = []

for i in tqdm(range(1, 323, 2)):
    try:
        roads = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2020/ROADS/tl_2020_13" + str(i).zfill(3) + "_roads.zip")
    except:
        print("No such file")
    for ind, row in roads.iterrows():
        road = row.geometry
        point = []
        for coord in road.coords:
            point.append([list(coord),row.LINEARID])
        points.append(point)

df = pd.DataFrame(columns=["x","y","road_id"])
x = []
y = []
id = []
for road in points:
    for coord in road:
        x.append(coord[0][0])
        y.append(coord[0][1])
        id.append(coord[1])

df.x = x
df.y = y
df.road_id = id

df.to_csv("./road_data/ga.csv")

json_file = json.dumps(points)
with open('./road_data/ga.json', 'w') as f:
    json.dump(points, f)