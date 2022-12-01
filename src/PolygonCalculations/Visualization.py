import matplotlib.pyplot as plt
from constants import api_key, central_coords
import gmplot
import folium
import geopandas as gpd
from shapely.geometry import Polygon

def plotPoints(point_list):
    xs = [point.x for point in point_list]
    ys = [point.y for point in point_list]



    plt.scatter(xs, ys)
    plt.show()

def plotPointsOnMap(coord_list, point_list):

    xs = [point.x for point in point_list]
    ys = [point.y for point in point_list]

    geometry = Polygon(coord_list)
    polygon = gpd.GeoDataFrame(index=[0], crs='epsg:4326', geometry=[geometry])

    points = gpd.GeoDataFrame(crs='epsg:4326', geometry=gpd.points_from_xy(xs, ys))

    m = folium.Map(central_coords, zoom_start=12, tiles='cartodbpositron')
    
    folium.GeoJson(polygon).add_to(m)
    folium.LatLngPopup().add_to(m)
    folium.GeoJson(points).add_to(m)
    
    m.save('map.html')