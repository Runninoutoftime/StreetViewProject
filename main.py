from GeoCalculation import GeoCalculation
from constants import file, api_key, central_coords, spacing, county_name, visualization_only
import json
from Visualization import *


# Create a GeoCalculation object for the specified county
# Generates a grid of points within the county
geo_calculation = GeoCalculation(county_name, file, spacing)
polygon_coords = geo_calculation.getCoords()
point_list = geo_calculation.gen_n_point_in_polygon(spacing, polygon_coords, 0.1)

# Exports the grid of points to a json file
json_point_list = [(pt.x, pt.y) for pt in point_list]
json_string = json.dumps(json_point_list)

if not visualization_only:
    with open('./point_lists/' + county_name + '.json', 'w', encoding='utf-8') as f:
        json.dump(json_string, f, ensure_ascii=False, indent=4)

plotPointsOnMap(polygon_coords, point_list)

# Now that I have the list of points
# I need to gather streetview images for each point based on the neaerest streetview image available
# Export points to a json file