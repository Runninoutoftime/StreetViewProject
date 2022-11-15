import pandas as pd
from random import sample

roads = pd.read_csv("./road_data/Clarke_county_roads.csv", index_col="Unnamed: 0")
points = list(zip(roads.x,roads.y,roads.road_id))
print(len(points))
points = sample(points, 2000) 

sampled_points = pd.DataFrame(columns=roads.columns)
for point in points:
    sampled_points.loc[len(sampled_points.index)] = point

sampled_points.to_csv("./sampled_road_points/clarke_county.csv")