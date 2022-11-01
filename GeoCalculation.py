import csv
from functools import partial
import numpy as np
import gmplot
import json
from shapely.geometry import Polygon, shape, Point
from shapely.prepared import prep
from shapely.ops import transform
import pyproj
from itertools import product
import matplotlib.pyplot as plt

# apikey=api_key

# gmap = gmplot.GoogleMapPlotter(33.9511687467, -83.3673290793, 14, apikey=apikey, map_type='hybrid')

class GeoCalculation:
    def __init__(self, geo_name, geo_json_file, spacing):
        self.geo_name = geo_name
        self.geo_json_file = geo_json_file
        self.spacing = spacing

    def getCoords(self):
        """
        Returns the coordinates of the points defining the polygon's perimeter

        Returns:
            list: list of coordinates (where each coordinate is a tuple of (x, y) in latitute and longitude)
        """
        file = open(self.geo_json_file, 'r')

        data = json.load(file)

        fulldata = data['fields']['geo_shape']['coordinates']
        coords = []
        for i in range(len(fulldata[0])):
            coords.append((fulldata[0][i][0], fulldata[0][i][1]))
        
        file.close()
        return coords

    def getArea(self):
        """
        Returns the area of the polygon in square meters

        Returns:
            float: area of the polygon in square meters
        """

        coords = self.getCoords()

        geometry =  {'type': 'Polygon', 'coordinates': [coords]}
        s = shape(geometry)
        proj = partial(pyproj.transform, pyproj.Proj(init='epsg:4326'), pyproj.Proj(init='epsg:3857'))
        newShape = transform(proj, s)
        return newShape.area

    def gen_n_point_in_polygon(self, n_point, polygon, tol = 0.1):
        """
        -----------
        Description
        -----------
        Generate n regular spaced points within a shapely Polygon geometry
        -----------
        Parameters
        -----------
        - n_point (int) : number of points required
        - polygon (shapely.geometry.polygon.Polygon) : Polygon geometry
        - tol (float) : spacing tolerance (Default is 0.1)
        -----------
        Returns
        -----------
        - points (list) : generated point geometries
        -----------
        Examples
        -----------
        >>> geom_pts = gen_n_point_in_polygon(200, polygon)
        >>> points_gs = gpd.GeoSeries(geom_pts)
        >>> points_gs.plot()
        """
        # Get the bounds of the polygon
        geometry =  {'type': 'Polygon', 'coordinates': [polygon]}
        polygon = shape(geometry)
        minx, miny, maxx, maxy = polygon.bounds    
        # ---- Initialize spacing and point counter
        spacing = polygon.area / n_point
        point_counter = 0
        # Start while loop to find the better spacing according to tolérance increment
        while point_counter <= n_point:
            print(point_counter)
            # --- Generate grid point coordinates
            x = np.arange(np.floor(minx), int(np.ceil(maxx)), spacing)
            y = np.arange(np.floor(miny), int(np.ceil(maxy)), spacing)
            xx, yy = np.meshgrid(x,y)
            print('test1')
            # ----
            pts = [Point(X,Y) for X,Y in zip(xx.ravel(),yy.ravel())]
            # ---- Keep only points in polygons
            points = [pt for pt in pts if pt.within(polygon)]
            print('test2')
            # ---- Verify number of point generated
            point_counter = len(points)
            spacing -= tol
        # ---- Return
        return points