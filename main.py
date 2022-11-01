from matplotlib import pyplot as plt
from GeoCalculation import GeoCalculation
from constants import file, api_key, coords, spacing



geo_calculation = GeoCalculation('Clarke County', file, spacing)

# print(geo_calculation.createGrid())
polygon_coords = geo_calculation.getCoords()
print('test')
point_list = geo_calculation.gen_n_point_in_polygon(10, polygon_coords, 0.1)

xs = [point.x for point in point_list]
ys = [point.y for point in point_list]
plt.scatter(xs, ys)
plt.show()