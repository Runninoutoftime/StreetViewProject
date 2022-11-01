from GeoCalculation import GeoCalculation
from constants import file, api_key, coords, spacing, county_name
import json



geo_calculation = GeoCalculation(county_name, file, spacing)

# print(geo_calculation.createGrid())
polygon_coords = geo_calculation.getCoords()
point_list = geo_calculation.gen_n_point_in_polygon(spacing, polygon_coords, 0.1)
# print("Num pts: ", len(point_list))
# Num of points generated on map using gen_n_point_in_polygon(Number, polygon_coords, tolerance of 0.1)
# 6 : 1179
# 5 : 813
# 4 : 523
# 3 : 291
# 2 : 131




json_point_list = [(pt.x, pt.y) for pt in point_list]
json_string = json.dumps(json_point_list)
print(json_string)

with open('./point_lists/' + county_name + '.json', 'w', encoding='utf-8') as f:
    json.dump(json_string, f, ensure_ascii=False, indent=4)

# Now that I have the list of points
# I need to gather streetview images for each point based on the neaerest streetview image available
# Export points to a json file