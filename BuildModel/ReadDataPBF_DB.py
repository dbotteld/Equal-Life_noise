### USES ROADS.SHP AND BUILDINGS.SHP AS INPUT FILES INSTEAD OF USING OSM
### THEREFOR GOES FASTER THAN ReadData.py
import numpy as np
import pandas as pd
import geopandas as gpd
import os
import re
import warnings
from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import rotate
from shapely.ops import transform
from shapely import wkb
import pyproj
import osmnx as ox
import matplotlib.pyplot as plt

import ml_indicators_DB as mli

model_folder = "I:\\ML_indicators\\branches\\AugustusLucas\\receiversnew\\"
shape_folder = "..\Roads\Belgium"

srid = 31370

# scenario identifies the region and traffic information that has been used
# 1: I:\ScenariosVlaanderenColin
# 2: I:\ScenariosVlaanderenLessTraffic
scenario = 1

level_cols = ["Leq", "LAeq", "LCeq", "LA01", "LA05", "LA10", "LA50", "LA90", "LA95", "LA99"]
event_count_cols = ["EN55", "EN60", "EN65", "EN70", "EN75", "EN80", "ENLA10", "ENLA50", "ENLA50+3", "ENLA50+10",
                    "ENLA50+15", "ENLA50+20", "ENLAeq+10", "ENLAeq+15"]
event_duration_cols = ["ET55", "ET60", "ET65", "ET70", "ET75", "ET80", "ETLA10", "ETLA50", "ETLA50+3", "ETLA50+10",
                       "ETLA50+15", "ETLA50+20", "ETLAeq+10", "ETLAeq+15"]
spectrum_cols = ["L_20", "L_25", "L_31", "L_40", "L_50", "L_63", "L_80", "L_100", "L_125", "L_160", "L_200", "L_250",
                 "L_315", "L_400", "L_500",
                 "L_630", "L_800", "L_1000", "L_1250", "L_1600", "L_2000", "L_2500", "L_3150", "L_4000", "L_5000",
                 "L_6300", "L_8000", "L_10000",
                 "L_12500", "L_16000", "L_20000"]
fluctuation_cols = ["SigmaAS", "SigmaCS", "LA10-LA90", "LC10-LC90", "IntRatio"]
ml_cols = ["LAeq", "LA50", "LA90", "LA10-LA90", "IntRatio", "ETLA50+3", "ETLA50+10", "CoG_125_8k", "N", "N50", "N90",
           "S05", "S10", "PSD_0_3", "PSD_3_6", "PSD_6_12", "PSD_12_18"]

epsg_4326 = pyproj.CRS('EPSG:4326')
epsg_target = pyproj.CRS("EPSG:%d" % srid)

project_4326_target = pyproj.Transformer.from_crs(epsg_4326, epsg_target, always_xy=True).transform
project_target_4326 = pyproj.Transformer.from_crs(epsg_target, epsg_4326, always_xy=True).transform

max_osm_dist = 2000
close_distance = 200
nb_angles = 36

print("READING ROADS")
roads_shp = gpd.read_file(shape_folder + "\\ROADS.shp")
print("READING BUILDINGS")
buildings_shp = gpd.read_file(shape_folder + "\\BUILDINGS.shp")

# RUN THE KERNELS AFTER THIS TO SAVE AFTER THIS KERNEL HAS FINISHED ITS RUN
import math

files_in_dir = os.listdir(model_folder)
df_model = gpd.GeoDataFrame()
input_data = []
output_features = []
wrongones = []
i = 0

for file in files_in_dir:
    if re.match(r"^indicators_model_rec\d+_PSD.csv$", file):
        print("reading file %s (%d/%d)" % (file, i, len(files_in_dir)))
        with warnings.catch_warnings(record=True):
            df_file = gpd.read_file(model_folder + "\\" + file)
            #df_file = df_file.iloc[:1, 1:] 
            df_file = df_file.iloc[:, 1:] # changed on dec 11, 2022 to include all (4) simulations
            df_file_records = df_file.shape[0]
            df_file["geometry"] = gpd.GeoSeries.from_wkt(df_file["geom"], crs="EPSG:%d" % srid)
            df_file.drop(columns=["geom"], inplace=True)
            if df_file.iloc[0]["S05"] == "" or df_file.iloc[0]["S10"] == "":
                print("No S05 or S10 information detected")
                i += 1
                continue
            df_model = df_model.append(df_file, ignore_index=True)

            receiver_point: Point = df_file.iloc[0]["geometry"]

            osm_point: Point = transform(project_target_4326, receiver_point)
            receiver_point = wkb.loads(wkb.dumps(receiver_point, output_dimension=2))
            print(osm_point.x, osm_point.y)

            polygon = receiver_point.buffer(max_osm_dist)
            df_roads = roads_shp.loc[roads_shp.intersects(polygon)].reset_index()
            df_buildings = buildings_shp.loc[buildings_shp.intersects(polygon)].reset_index()

            df_roads = df_roads.rename(columns={"TYPE": 'highway', "LANES": 'lanes', "ONEWAY": 'oneway'})

            if "lanes" not in df_roads:
                wrongones.insert(0, i)
                df_model = df_model.iloc[:-df_file_records]
                print("No lanes information detected")
                i += 1
                continue
            if "oneway" not in df_roads:
                wrongones.insert(0, i)
                df_model = df_model.iloc[:-df_file_records]
                print("No oneway information detected")
                i += 1
                continue
            df_roads = mli.add_traffic_intensity(df_roads, scenario)

            polygon = receiver_point.buffer(close_distance)
            df_close_roads = df_roads.loc[df_roads.intersects(polygon)]
            df_close_buildings = df_buildings.loc[df_buildings.intersects(polygon)]
            pointinbuild = []
            for key, geom in df_close_buildings["geometry"].items():
                ptinbuild = Point(receiver_point.x, receiver_point.y).within(geom)
                pointinbuild.append(ptinbuild)
            if not any(pointinbuild):
                print("Point is not within a building")
            else:
                print("Point is inside of a building")
                wrongones.insert(0, i)
                df_model = df_model.iloc[:-df_file_records]
                i += 1
                continue  # go on to next receiver
            pointinroad = []
            for key, geom in df_close_roads["geometry"].items():
                ptinroad = Point(receiver_point.x, receiver_point.y).within(geom)
                pointinroad.append(ptinroad)
            if not any(pointinroad):
                print("Point is not on a road")
            else:
                print("Point is on a road")
                wrongones.insert(0, i)
                df_model = df_model.iloc[:-df_file_records]
                i += 1
                continue  # go on to next receiver
            # Building densities by angle
            output_array = []
            output_array += list(mli.get_building_surface_by_angle(receiver_point, close_distance, nb_angles, df_close_buildings).values())
            # Closest building by angle
            build_in_dist = list(mli.get_closest_object_distance_by_angle(receiver_point, close_distance, nb_angles, df_close_buildings).values())
            new_build_in_dist = [1 / (a+1) for a in build_in_dist]
            for b in range(len(new_build_in_dist)):
                if new_build_in_dist[b] == -1:
                    new_build_in_dist[b] = 0
            output_array += new_build_in_dist
            # Closest roads with traffic intensities between min_lv - max_lv by angle
            min_lv = 0
            for max_lv in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 99999]:
                filtered_lv_roads = df_close_roads.loc[(df_close_roads["lv_d"] > min_lv) & (df_close_roads["lv_d"] <= max_lv)]
                # if len(filtered_lv_roads) == 0
                # output_array += [-1]
                road_in_dist = list(mli.get_closest_object_distance_by_angle(receiver_point, close_distance, nb_angles, filtered_lv_roads).values())
                new_road_in_dist = [1 / (x+1) for x in road_in_dist]
                for k in range(len(new_road_in_dist)):
                    if new_road_in_dist[k] == -1:
                        new_road_in_dist[k] = 0
                output_array += new_road_in_dist
                min_lv = max_lv
            # Adding of total building surface between distances & total road length of certain traffic intensities between the distances
            min_dist = 200
            for max_dist in [300, 500, 1000, 2000]:
                output_array += [mli.get_building_surface_between_distances_fast(receiver_point, min_dist, max_dist, df_buildings)]
                min_lv = 200
                for max_lv in [400, 800, 1600, 99999]:
                    filtered_lv_roads = df_roads.loc[(df_roads["lv_d"] >= min_lv) & (df_roads["lv_d"] < max_lv)]
                    # if len(filtered_lv_roads) == 0:
                    # output_array += [0.0]
                    output_array += [mli.get_road_length_between_distances_fast(receiver_point, min_dist, max_dist, filtered_lv_roads)]
                    min_lv = max_lv
                min_dist = max_dist
            for newrow in range(df_file_records): # changed on Dec 11 2022 to repeat for each calculation of indicators
                input_data.append(output_array)
    i += 1

output_features = df_model[ml_cols].to_numpy(dtype=np.float32)
input_data = np.array(input_data)
print(input_data.shape)
print(output_features.shape)

#RUN THESE 2 KERNELS TO SAVE THE DATA TO A NUMPY FILE
output_features = df_model[ml_cols].to_numpy(dtype=np.float32)
output_geom = df_model["geometry"]
input_data = np.array(input_data)

#SAVES THE NUMPY FILES, CHANGE STRING TO DESIRED NAME
if scenario == 1:
	np.save("output_features_def4001-5000PBF_sc1.npy", output_features)
	np.save("input_data4001-5000PBF_sc1.npy", input_data)
	np.save("input_datageom4001-5000PBF_sc1.npy", output_geom)
elif scenario == 2:
	np.save("output_features_def3001-4000PBF_sc2.npy", output_features)
	np.save("input_data3001-4000PBF_sc2.npy", input_data)
	np.save("input_datageom3001-4000PBF_sc2.npy", output_geom)
