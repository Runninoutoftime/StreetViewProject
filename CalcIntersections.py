import pandas as pd
import geopandas as gpd
import json
from shapely.geometry import Polygon, shape, Point, LineString, MultiPoint, MultiLineString
from tqdm import tqdm
import os
from random import sample
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from scipy.ndimage.filters import gaussian_filter
import numpy as np

class CalcIntersections:
    def __init__(self, state_fp):
        self.state_fp = state_fp
        if not os.path.exists("./point_lists/FIP_" + str(state_fp)):
            os.makedirs("./point_lists/FIP_" + str(state_fp))
        if not os.path.exists("./road_data/FIP_" + str(state_fp)):
            os.makedirs("./road_data/FIP_" + str(state_fp))    
        if not os.path.exists("./intersections/FIP_" + str(state_fp)):
            os.makedirs("./intersections/FIP_" + str(state_fp))
        if not os.path.exists("./intersections/FIP_" + str(state_fp)):
            os.makedirs("./intersections/FIP_" + str(state_fp))             
        if not os.path.exists("./sampled_road_points/FIP_" + str(state_fp)):
            os.makedirs("./sampled_road_points/FIP_" + str(state_fp))
    def getCounties(self):
        """
        Downloads and returns the bounding shape coordinates for all counties within a given state
        
        Returns
        -------
        - county_geo (list) : list of points for county geometry
        """
        counties = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2022/COUNTY/tl_2022_us_county.zip")
        for ind, row in counties.iterrows():
            county_name = row.NAME
            county_geo = []
            if row.STATEFP == str(self.state_fp):
                try:
                    coords = row.geometry.exterior.coords
                    for coord in coords:
                        county_geo.append(list(coord))
                except:
                    for polygon in row.geometry:
                        coords = polygon.exterior.coords
                        for coord in coords:
                            county_geo.append(list(coord))
                
                with open('./point_lists/FIP_' + str(self.state_fp) + '/' + county_name + '_' + 'county.json', 'w') as f:
                    json.dump(county_geo, f)
        return county_geo

    def getRoads(self, min_county_fp, max_county_fp, fp_step):
        """
        Downloads and returns a list of coordinates and id's for all roads within a given state
        
        Parameters
        ----------
        - min_county_fp (int) : starting value for county FIPS
        - max_county_fp (int) : maximum value for county FIPS
        - fp_step (int) : interval of county FIPS within state
        Returns
        -------
        - roads (list) : list of roads (each road is a list of coordinates as lists of x and y), and their road ID's
        """
        
        roads = []
        for i in tqdm(range(min_county_fp, max_county_fp, fp_step)):
            try:
                state_roads = gpd.read_file("https://www2.census.gov/geo/tiger/TIGER2020/ROADS/tl_2020_" + str(self.state_fp) + str(i).zfill(3) + "_roads.zip")
            except:
                print("No such file")
            for ind, row in state_roads.iterrows():
                row_road = row.geometry
                road = []
                for coord in row_road.coords:
                    road.append(list(coord))
                roads.append([road,row.LINEARID])
        
        with open("./road_data/FIP_" + str(self.state_fp) + "/full_state.json", 'w') as f:
            json.dump(roads, f)
        
        return roads
    
    def sortRoads(self, county_names):
        """
        Sorts through roads of entire state to make json of roads for each county
        
        Parameters
        ----------
        - county_names (list) : list of counties to search for roads
        """

        road_file = open("./road_data/FIP_" + str(self.state_fp) + "/full_state.json", 'r')
        roads = json.load(road_file)
        road_file.close()

        for county_name in county_names:
            print(county_name)
            point_file = open("./point_lists/FIP_" +str(self.state_fp) + "/" + county_name + ".json", 'r')
            coords = json.load(point_file)
            point_file.close()

            county_geometry =  {'type': 'Polygon', 'coordinates': [coords]}
            county_polygon = shape(county_geometry)
            road_in_county = []
            for road in tqdm(roads):
                points_in_county = []
                in_county = False
                for point in road[0]:
                    shp_point = Point(point[0], point[1])
                    if (county_polygon.contains(shp_point)):
                        in_county = True
                        points_in_county.append(point)
                if (in_county):
                    road_in_county.append([points_in_county, road[1]])
            with open("./road_data/FIP_" + str(self.state_fp) + "/" + county_name+ ".json", 'w') as f:
                json.dump(road_in_county, f)

    def findIntersections(self, county_names):
        """
        Searches through road of county to make list of intersections in the county
        
        Parameters
        ----------
        - county_names (list) : list of counties to search for intersections
        """
        for county_name in county_names:
            print(county_name)
            road_file = open("./road_data/FIP_" + str(self.state_fp) + "/" + county_name+ ".json", 'r')
            roads = json.load(road_file)
            road_lines = []
            road_file.close()
            for road in roads:
                road_line = [tuple(coord) for coord in road[0]]
                if len(road_line) > 1:
                    road_lines.append(LineString(road_line))          
            intersections = []
            for ind in tqdm(range(len(road_lines))):
                for comp_ind in range(ind + 1, len(road_lines)):
                    intersection = road_lines[ind].intersection(road_lines[comp_ind])
                    if(intersection):
                        if (isinstance(intersection, Point)):
                            intersections.append(intersection.coords[0])
                        #elif (isinstance(intersection, LineString)):
                        #    for coord in intersection.coords:
                        #        intersections.append(coord)
                        #elif (isinstance(intersection, MultiLineString)):
                            #for line in intersection:
                                #for coord in line.coords:
                                    #intersections.append(coord)
                        elif (isinstance(intersection, MultiPoint)):
                            points = [list(x.coords) for x in list(intersection)]
                            for point in points:
                                intersections.append(point[0])
            with open("./intersections/FIP_" + str(self.state_fp) + "/" + county_name+ ".json", 'w') as f:
                json.dump(intersections, f)          
        
    def samplePoints(self, county_names, n=2000):
        """
        Samples n points from list of intersections within the county
        
        Parameters
        ----------
        - county_names (list) : list of counties to search for roads
        - n (int) : number of intersections to sample
        """

        for county_name in county_names:
            intersection_file = open("./intersections/FIP_" + str(self.state_fp) + "/" + county_name+ ".json", 'r')
            intersections = json.load(intersection_file)   
            intersection_file.close()             
            points = sample(intersections, n)
            sample_df = pd.DataFrame(columns=["x","y"])
            for i, point in enumerate(points):
                sample_df.loc[i] = {"x": point[0], "y": point[1]}

            sample_df.to_csv("./sampled_road_points/FIP_" + str(self.state_fp) + "/" + county_name + ".csv", index=False)

    def heatMap(self, county_names):
        if not os.path.exists("./heat_maps/FIP_" + str(self.state_fp)):
            os.makedirs("./heat_maps/FIP_" + str(self.state_fp))

        for county_name in county_names:    
            road_file = open("./road_data/FIP_" + str(self.state_fp) + "/" + county_name + ".json", 'r')
            roads = json.load(road_file)
            road_file.close()
            sample_roads = pd.read_csv("./sampled_road_points/FIP_" + str(self.state_fp) + "/" + county_name + ".csv")
            figure(figsize=(50, 50), dpi=80)
            heatmap, xedges, yedges = np.histogram2d(sample_roads.x  , sample_roads.y, bins=3000)
            heatmap = gaussian_filter(heatmap, sigma=32) #tweak the sigma value a bit for different visual results
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]] 
            plt.imshow(heatmap.T, extent=extent,origin='lower')
            for road in roads:
                x = [i[0] for i in road[0]]
                y = [i[1] for i in road[0]]
                plt.plot(x,y, color="black")
            plt.savefig("./heat_maps/FIP_" + str(self.state_fp) + "/" + county_name + "_heatmap.png")


