from tensorflow.keras.models import load_model
import numpy as np
from pandas import read_csv, isna, DataFrame
from geopandas import GeoDataFrame, GeoSeries
import pickle
import os
import re
import warnings
from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import rotate
from shapely.ops import transform
from shapely.wkt import loads
from shapely import wkb
from shapely import wkt
import pyproj
import osmnx as ox
import matplotlib.pyplot as plt
from copy import copy
import warnings
import time
import sys
from multiprocessing import *
import json

import ml_indicators_EB2 as mli
from GroupScaler import GroupScaler
from streets_parallel import *
from geo_operations import *


from itertools import islice

def chunks(data, SIZE=100000):
   it = iter(data)
   for i in range(0, len(data), SIZE):
      yield {k:data[k] for k in islice(it, SIZE)}


#python UseModel\ml_indicators_for_area.py UseModel\STARS.json  

ox.settings.timeout = 1000

# ======= set area input and output files here =======
json_path = sys.argv[1]
#json_path = "UseModel\ghent_antwerp_200dwellings.json"


with open(json_path, "r") as file:
    conf_data = json.load(file)

# conf_data = json.loads(json_path) 
# ======= set area input and output files here =======
region= conf_data["region"]

file_receivers = conf_data["recievers"]
srid_of_receiver = conf_data["srid recievers"] #use 4326 if in OSM lat lon; use 3857 for google maps
PID = conf_data["PID"] 
file_indicators_backup = conf_data["indicators backup"]
file_geom_backup = conf_data["geom backup"]
file_recievers_backup = conf_data["recievers backup"]
file_indicators_save = conf_data["indicators save"]
file_geom_save = conf_data["geom save"]
file_features_save = conf_data["features save"]
file_hourlyshape_save = conf_data["hourlyshape save"]
file_dwellingshape_save = conf_data["dwellingshape save"]
file_traffic_save = conf_data["traffic save"]
read_EB = conf_data["read EB"]
building_nearby = conf_data["building nearby"] # this saves time if some of the receiver points are not near a house and you are looking for population exposure
preload_buildings = conf_data["preload buildings"]
file_buildings_save = conf_data["buildings save"] #for reference and mapping purposes
#unique_index = conf_data["unique index"]

# resolve issues with multiple installs of pyproj
print(pyproj.datadir.get_data_dir())
pyproj.datadir.set_data_dir(r'C:\Users\ldkoninc\.conda\envs\ml_environment\Library\share\proj')
print(pyproj.datadir.get_data_dir())

# local Cartesian projection to use
#srid = 31370        #FLANDERS
# srid = 2062         #BARCELONA
#srid = 31287        #AUSTRIA
srid = conf_data["srid"]         #Sweden

buffer = conf_data["buffer"] # Buffer of 10 km for EB calculations 


if srid == srid_of_receiver:
    point_in_Cartesian = True
else:
    point_in_Cartesian = False

# ======= end of local settings ======================

# trained model input files
file_model = 'UseModel/model_6'
file_input_scaler = 'UseModel/CustomGroupInputScaler.pkl'
file_output_scaler = 'UseModel/CustomOutputScalerPBF2.pkl'
# model parameters that are related to the trained ML, should not be changed
max_osm_dist = 2000
close_distance = 200
building_distance = 20
nb_angles = 36

crs_of_receiver = ("EPSG:%d" % srid_of_receiver)
epsg_4326 = pyproj.CRS('EPSG:4326')
epsg_target = pyproj.CRS("EPSG:%d" % srid)

project_4326_target = pyproj.Transformer.from_crs(epsg_4326, epsg_target, always_xy=True).transform
project_target_4326 = pyproj.Transformer.from_crs(epsg_target, epsg_4326, always_xy=True).transform

def test_if_in_building(x_point:float, y_point:float, preloaded_buildings:bool, direction:int = -1):
    """
    This function tests if a point is within a building and moves it outside if it is.
    
    Parameters:
    x_point (float): x coordinate of the point.
    y_point (float): y coordinate of the point.
    preloaded_buildings (bool): flag indicating if building data is preloaded.
    direction (int, optional): direction to move the point, with default value -1.
    
    Returns:
    list: A list of Points outside of buildings.
    """
    point_target: Point = transform(project_4326_target, Point(x_point, y_point))
    if preloaded_buildings:
        polygon = point_target.buffer(25)
        df_buildings = df_all_buildings.loc[df_all_buildings.intersects(polygon)].reset_index()
    else:
        df_buildings = ox.geometries_from_point((y_point, x_point), dist=25, tags={"building": True})
        df_buildings = df_buildings.to_crs("EPSG:%d" % srid)
    pointinbuild = []
    newpointlist = []
    if len(df_buildings.index) == 0:
        newpointlist.append(Point(x_point, y_point))
        return newpointlist
    for key, geom in df_buildings["geometry"].items():
        ptinbuild = point_target.within(geom)
        pointinbuild.append(ptinbuild)
    if not any(pointinbuild):
        newpointlist.append(Point(x_point, y_point))
        return newpointlist
    else:
        # print("  Point in building (moving point...)")
        for i in range(4):
            x_point = point_target.x
            y_point = point_target.y
            if i==0 and (direction == -1 or direction == 0):
                x_point += 5
            elif i==1 and (direction == -1 or direction == 1):
                x_point -= 5
            elif i==2 and (direction == -1 or direction == 2):
                y_point += 5
            elif i==3 and (direction == -1 or direction == 3):
                y_point -= 5
            else:
                continue
            point_osm: Point = transform(project_target_4326, Point(x_point, y_point))
            x_point = point_osm.x
            y_point = point_osm.y
            newpointlist.extend(test_if_in_building(x_point, y_point, preloaded_buildings, direction=i))
        return newpointlist

def init_features_for_ML(df1:GeoDataFrame, df2:GeoDataFrame) -> None:
    """
    This function is used to copy several variables when initializing a pool of threads. 
    """
    global df_roadsEB
    global df_all_buildings
    df_roadsEB = df1
    df_all_buildings = df2


def features_for_ML(receiver_point,i,preloaded_buildings,pre_calculated_traffic):
    """
    This function computes features for machine learning (ML) based on input data including the receiver point, preloaded buildings and pre-calculated traffic. 
    It calculates the traffic intensity and various building-related features such as building density by angle, closest building by angle, and closest road. 
    The input data is transformed and the required data is extracted based on the conditions provided in the code. The output of the function is an array of 
    building-related features.

    Input:
    receiver_point: Point object representing the location of the receiver
    i (int): a counter variable
    preloaded_buildings (Boolean): whether the buildings data is preloaded or not
    pre_calculated_traffic (Boolean): whether the traffic data is pre-calculated or not

    Output:
    output_array (list): a list of building-related features for the given receiver point
    """ 

    input_data = np.array([]).reshape(0, 452)
    output_geom = []
    with warnings.catch_warnings(record=True):
        start1 = time.time()
#        print("READING POINT ", i)
        if point_in_Cartesian:
            osm_point: Point = transform(project_target_4326, receiver_point)
        else:
            osm_point: Point = receiver_point
            receiver_point: Point = transform(project_4326_target, receiver_point)
        receiver_point = wkb.loads(wkb.dumps(receiver_point, output_dimension=2))

        x_point = osm_point.x
        y_point = osm_point.y
        #print([x_point, y_point]) #debug only
        pointlist = test_if_in_building(x_point, y_point, preloaded_buildings)
        early_stop = False
        for subpoint in pointlist:
            if early_stop:
                continue #once it was decided for one of the shifted points that calculation is useless, we stop all
            x_point = subpoint.x
            y_point = subpoint.y
            receiver_point: Point = transform(project_4326_target, Point(x_point, y_point))

            start2 = time.time()
            if read_EB:
                polygon = receiver_point.buffer(max_osm_dist) #is this correct
                df_roads = df_roadsEB.loc[df_roadsEB.intersects(polygon)].reset_index()
            else:
                df_roads = ox.geometries_from_point((y_point, x_point), dist=max_osm_dist, tags={"highway": True})
                df_roads = df_roads.to_crs("EPSG:%d" % srid)
                df_roads = complete_roads(df_roads)
#            print('  Load roads: ', time.time() - start2)
            if len(df_roads.index) == 0:
                print("  No roads")
                #i += 1
                early_stop = True # when there are no roads within 2 km we skip calculations
                continue

            start2 = time.time()
            
            if preloaded_buildings:
                polygon = receiver_point.buffer(max_osm_dist)
                df_buildings = df_all_buildings.loc[df_all_buildings.intersects(polygon)].reset_index()
            else:
                df_buildings = ox.geometries_from_point((y_point, x_point), dist=max_osm_dist, tags={"building": True})
                df_buildings = df_buildings[df_buildings.geom_type=='Polygon']
                df_buildings = df_buildings.to_crs("EPSG:%d" % srid)
            #print('  Load buildings: ', time.time() - start2)

            if building_nearby:
                polygon = receiver_point.buffer(building_distance)
                df_nearby_building = df_buildings.loc[df_buildings.intersects(polygon)]
                if len(df_nearby_building.index) == 0:
                    print("  No building nearby")
                    #i += 1
                    early_stop = True #if there is no building within 200m it is not useful to calculate
                    continue 
            if not pre_calculated_traffic:
                traffic = []
                start2 = time.time()
                for j in range(len(df_roads)):
                    traffic.append(mli.get_traffic_intensity(df_roads, j))
                df_roads["lv"] = traffic
                #print('  Calculate traffic intensity: ', time.time() - start2)
            polygon = receiver_point.buffer(close_distance)
            df_close_roads = df_roads.loc[df_roads.intersects(polygon)]
            output_array = []
            if len(df_close_roads.index) == 0:
                print("  No close roads")
                output_array = np.zeros((24,432))
                #i += 1
                #continue # why would we stop and ignore far away roads?
            else:
                df_close_buildings = df_buildings.loc[df_buildings.intersects(polygon)]
        # Building densities by angle
                output_array += list(mli.get_building_surface_by_angle(receiver_point, close_distance, nb_angles, df_close_buildings).values())
        # Closest building by angle
                build_in_dist = list(mli.get_closest_object_distance_by_angle(receiver_point, close_distance, nb_angles, df_close_buildings).values())
                new_build_in_dist = [1 / (a+1) for a in build_in_dist]
                for b in range(len(new_build_in_dist)):
                    if new_build_in_dist[b] == -1:
                        new_build_in_dist[b] = 0
                output_array += new_build_in_dist
                output_array = np.tile(output_array, (24, 1))
        # Closest roads with traffic intensities between min_lv - max_lv by angle
                min_lv = 0
                start2 = time.time()
                stack = np.stack(df_close_roads["lv"])
                for max_lv in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 99999]:
                    sub_output_array = np.zeros((24, 36))
                    for hour in range(24):
                        maskindex = stack[:, hour]
                        mask = (maskindex >= min_lv) & (maskindex < max_lv)
                        filtered_lv_roads = df_close_roads.loc[mask]#[(df_close_roads["lv"][hour] > min_lv) & (df_close_roads["lv"][hour] <= max_lv)]
                # if len(filtered_lv_roads) == 0
                # output_array += [-1]
                        road_in_dist = list(mli.get_closest_object_distance_by_angle(receiver_point, close_distance, nb_angles, filtered_lv_roads).values())
                        new_road_in_dist = [1 / (x+1) for x in road_in_dist]
                        for k in range(len(new_road_in_dist)):
                            if new_road_in_dist[k] == -1:
                                new_road_in_dist[k] = 0
                        sub_output_array[int(hour)] = new_road_in_dist
                    output_array = np.concatenate((output_array, sub_output_array), axis=1)
                    min_lv = max_lv
        # Adding of total building surface between distances & total road length of certain traffic intensities between the distances
            #print("  Calculate close objects: ", time.time() - start2)
            start2 = time.time()
            stack = np.stack(df_roads["lv"])
            min_dist = 0
            for max_dist in [200, 400, 800, 1600]:
                output_array = np.c_[output_array, np.ones(24)*mli.get_building_surface_between_distances(receiver_point, min_dist, max_dist, df_buildings)]
                min_lv = 200
                for max_lv in [400, 800, 1600, 99999]:
                    sub_output_array = np.zeros(24)
                    for hour in range(24):
                        maskindex = stack[:, hour]
                        mask = (maskindex >= min_lv) & (maskindex < max_lv)
                        filtered_lv_roads = df_roads.loc[mask]
                    #if len(filtered_lv_roads) == 0:
                    # output_array += [0.0]
                        sub_output_array[int(hour)] = mli.get_road_length_between_distances(receiver_point, min_dist, max_dist, filtered_lv_roads)
                    output_array = np.c_[output_array, sub_output_array]
                    min_lv = max_lv
                min_dist = max_dist
            #print("  Calculate road length: ", time.time() - start2)
            input_data = np.vstack([input_data, output_array])
            output_geom.append([i, Point(x_point, y_point)])
        print("  Total time: %s for POINT %s" % (time.time() - start1,i))
    return [input_data, output_geom]
    
def SDI(PSDs:float) -> float:
    """ 
    Calculate sleep disturbance index for one night from probability of sleep disturbance

    Input:
    PSDs (float): probability of sleep disturbance

    Output (float): sleep disturbance index 
    """
    # time is calculated in quarters starting at 17:00 

    theSDI = np.zeros(4)
    sleep_start = np.array([0, 7, 13, 19])  # age dependent sleep start measured from 17:00
    sleep_weigth = np.ones(4*14)
    for i in range(4*12, 4*14):
        sleep_weigth[i] = 1 + (i-4*12.0)/(4*14.0-4*12.0)

    PSD_nat = 0.05
    PSDs_weighted = np.multiply(np.maximum(0.0, (PSDs-PSD_nat)), sleep_weigth)

    for i in range(0, 4):
        theSDI[i] = np.sum(PSDs_weighted[i, sleep_start[i]:-1]) / (4*14.0-sleep_start[i])

    return theSDI
    #return [('SDI_0_3', theSDI[0]), ('SDI_3_6', theSDI[1]), ('SDI_6_12', theSDI[2]), ('SDI_12_18', theSDI[3])]

def save_indicators(recievers, input_data, output_geom, file_recievers, file_input, file_geom, counter = 0):
    try: 
        recievers.to_csv(file_recievers, mode="a", header=not os.path.exists(file_recievers))

        input_data = np.array(input_data)
        np.save(file_input, input_data, allow_pickle=True) #save the new

        output_geom = np.array(output_geom)
        np.save(file_geom, output_geom, allow_pickle=True) #save the new
    except: 
        if counter <= 5: 
            time.sleep(1)
            counter += 1
            save_indicators(recievers, input_data, output_geom, file_recievers, file_input, file_geom, counter)
        else:
            print("error while saving the file")
    return





if __name__ == "__main__":

    start0 = time.time()
    # get the receiver coordinates
    file = read_csv(file_receivers)
    file['geometry'] = file['geometry'].apply(loads)
    file = GeoDataFrame(file, geometry='geometry', crs=srid_of_receiver)

    data = file['geometry']
    data.reset_index(drop=True, inplace=True)
    print(file)
    PGTPI = file[PID] #the label should be defined in json to make this more flexible
    PGTPI.reset_index(drop=True, inplace=True)
    Npoints = len(data)
 

    # Read backup data if it exists 
    if os.path.exists(file_recievers_backup): 
        file_calculated = read_csv(file_recievers_backup, index_col=0)
        print("the following is already calculated:")
        print(file_calculated)
        
        # delete backup data from file so we do not calculate it again
        file = file[~file[PID].isin(file_calculated[PID])].dropna()

        input_data = np.load(file_indicators_backup, allow_pickle=True)
        output_geom = list(np.load(file_geom_backup, allow_pickle=True))
    else: 
        input_data = np.array([]).reshape(0, 452)
        output_geom = []
    
    if len(file)>1: #changed to 1 because no clustering is possible if only one element left
        # file_part = create_partition(file, 10000, srid)
        file_part, cells = create_part(file, crs = srid, max_cluster_size=25000**2)   # create clustered partition 
        if np.isnan(file_part["index_right"].max()):
            N_cells = -1
        else:
            N_cells = int(file_part["index_right"].max())+1
        print(file_part)
    else: 
        file_part = {"index_right": np.array([-1])}
        N_cells = -1

    
    if N_cells != -1:
        for cell_number in range(N_cells):
        # for cell_number in range(20):                           # debug
            start0 = time.time()
            file = file_part[file_part["index_right"]==cell_number]
            cell = cells.iloc[cell_number]['geometry']
            if not file.empty:
                # get EB file if it exists
                # southwest_EB, northeast_EB, southwest_osm_EB, northeast_osm_EB = create_bounding_box(file.total_bounds, buffer, point_in_Cartesian, project_4326_target, project_target_4326)
                # print(f"southwest_EB = {southwest_EB}")

                data_cell = file['geometry']
                data_cell.reset_index(drop=True, inplace=True)
                PGTPI_cell = file[PID] 
                PGTPI_cell.reset_index(drop=True, inplace=True)

                Npoints_cell = len(data_cell)
                Nchunk = min(20,int(Npoints_cell/10))

                print(f"calculating cell {cell_number} of {N_cells}")
                if read_EB:
                    # try:
                    df_roadsEB = get_streets_per_polygon([cell], network_type='drive', intersection_clos=False,  street_betw=True, street_sin=False, path = file_traffic_save)
                    for i in range(len(df_roadsEB)):
                        if (df_roadsEB['lanes'][i]) != 'nan':
                            if str(df_roadsEB['lanes'][i])[0] == '[':
                                df_roadsEB['lanes'][i] = min(list(map(int, df_roadsEB['lanes'][i])))
                    df_roadsEB = df_roadsEB.to_crs("EPSG:%d" % srid)
                    print("EB done")
                    # except:
                    #     df_roadsEB = GeoDataFrame()
                    #     for key in required_keys:
                    #         if key not in df_roadsEB.keys():
                    #             df_roadsEB[key] = 'nan'
                    #     print("EB ignored due to error")

            
                # decide to precalculate traffic in case of edge betweenness
                # get bounding box and perform some checks
                southwest, northeast, southwest_osm, northeast_osm = create_bounding_box(file.total_bounds, max_osm_dist, point_in_Cartesian, project_4326_target, project_target_4326)
                    
                pre_calculated_traffic = False    
                area_bounds = (northeast.x-southwest.x)*(northeast.y-southwest.y)
                #print(area_bounds)
                if (Npoints_cell*max_osm_dist*max_osm_dist > 4*area_bounds) and read_EB: #this means traffic calculation on same roads
                    traffic = []
                    for j in range(len(df_roadsEB)):
                        traffic.append(mli.get_traffic_intensity(df_roadsEB, j))
                    df_roadsEB["lv"] = traffic
                    pre_calculated_traffic = True
                    
                # preload buildings if required
                df_all_buildings = []
                if preload_buildings:
                    if (northeast.x-southwest.x)<150000 and (northeast.y-southwest.y)<150000: #make sure the preloaded area is not too big (receivers are together)
                        north = northeast_osm.y
                        south = southwest_osm.y
                        east = northeast_osm.x
                        west = southwest_osm.x
                        df_all_buildings = ox.geometries.geometries_from_polygon(cell, tags={"building": True})
                        df_all_buildings = GeoDataFrame(df_all_buildings, geometry='geometry', crs='EPSG:4326')
                        df_all_buildings = df_all_buildings[df_all_buildings.geom_type=='Polygon']
                        df_all_buildings = df_all_buildings.to_crs("EPSG:%d" % srid)
                        # df_all_buildings.to_csv(file_buildings_save)
                        preloaded_buildings = True
                    else:
                        print("Buildings not preloaded due to large bounding box")
                        print("width %s, height %s)" % (northeast.x-southwest.x, northeast.y-southwest.y))
                        preloaded_buildings = False
                else:
                    preloaded_buildings = False

                print("Preprocessing time: %s" % (time.time() - start0))
                
                # === calculate input features for ML model ===
                # use pool to extract features in parallel
                if Nchunk>=1: 
                    from multiprocessing import Pool, Lock
                    print("starting pool")
                    with Pool(processes = 12, initializer = init_features_for_ML, initargs=(df_roadsEB, df_all_buildings)) as pool:
                        # Npoints_cell = 4 #debug
                        items = [(data_cell[i], PGTPI_cell[i], preloaded_buildings, pre_calculated_traffic) for i in range(Npoints_cell)]
                        #items = [(data_cell[i], i, preloaded_buildings, pre_calculated_traffic) for i in range(Npoints_cell)]
                        print("pool started")
                        results = pool.starmap(features_for_ML, items, chunksize=Nchunk)
                        
                else:
                    results = []
                    for i in range(Npoints_cell):
                        print(f"point {i} of {Npoints_cell}")
                        results.append(features_for_ML(data_cell[i], PGTPI_cell[i], preloaded_buildings, pre_calculated_traffic))

                for result in results:
                    in_array = result[0]
                    geom = result[1]
                    input_data = np.vstack([input_data, in_array])
                    if geom: 
                        for igeom in range(len(geom)):
                            output_geom.append(geom[igeom])
                # make a backup so we can restart later if the program is interupted 
                save_indicators(file, input_data, output_geom, file_recievers_backup, file_indicators_backup, file_geom_backup)

            
    # === Apply ML model for indicators ===
    
    model = load_model(file_model)
    scaler_input = pickle.load(open(file_input_scaler,'rb'))
    scaler_output = pickle.load(open(file_output_scaler,'rb'))
    # reshape the input data for ML
    input_data = np.array(input_data)
    # Npoints = len(input_data)
    output_geom_a = np.array(output_geom) 
    #print(input_data.shape)
    #print(output_geom_a.shape)
    input_data_scaled = scaler_input.transform(input_data)

    data_length = len(input_data_scaled)
    imagedata = np.zeros([data_length,22,36])
    for y in range(data_length):
        for j in range(22):
            i = j
            if j%2 == 1:
                i = 1
            elif j == 0:
                i = 0
            else:
                i = int(j/2)+1
            imagedata[y][j]=input_data_scaled[y][0+(i*36):36+(i*36)]

    distdata = np.zeros([data_length,20])
    for p in range(data_length):
        distdata[p] = input_data_scaled[p][432:452]

    output_data_scaled = model.predict([imagedata, distdata])
    output_data = scaler_output.inverse_transform(output_data_scaled)
    #print(output_data.shape)

    output_data = output_data.reshape((int(len(output_data)/24)), 24, 17)
    #print(output_data.shape)

    np.save(file_indicators_save, output_data) # for backward compatibility
    np.save(file_geom_save, output_geom_a)
    
    #crs_of_receiver = 'EPSG:4326'
    ml_cols = ["LAeq", "LA50", "LA90", "LA10-LA90", "IntRatio", "ETLA50+3", "ETLA50+10", "CoG_125_8k", "N", "N50", "N90", "S05", "S10", "PSD_0_3", "PSD_3_6", "PSD_6_12", "PSD_12_18"]
    dictionairy = {ml_cols[j]+"@"+str(i)+"h": output_data[:, i, j] for j in range(17) for i in range(24)}
    dictionairy["geometry"] = output_geom_a[:,1]
    dictionairy[PID] = output_geom_a[:,0].astype(float) # changed for Walnuts 
#    dictionairy["dwelling"] = output_geom_a[:,0]
    gdf = GeoDataFrame(dictionairy, crs=crs_of_receiver)
    gdf.to_file(file_hourlyshape_save)
    
    # insert code to split geopandas in multiple chuncks
    ss = 0
    for n_ch, chunck in enumerate(chunks(dictionairy, 100000)):
        file_hourlyshape_save_chunck = file_hourlyshape_save + str(n_ch).zfill(3)
        gdf = GeoDataFrame(chunck, crs=crs_of_receiver)
        gdf.to_file(file_hourlyshape_save_chunck)

    
    # === post-process to dwelling indicators ===    
    dwelling_cols = ["SDI_0_3", "SDI_3_6", "SDI_6_12", "SDI_12_18", "ARP", "Lnight", "Lden"]
    dwelling_indicators = np.zeros((output_data.shape[0],len(dwelling_cols)))
    # first calculate per shifted receiver point
    # PSD to SDI
    PSDs = np.zeros((4, 4*14))
    for location in range(output_data.shape[0]):
        for i in range(17,24):
            PSDs[0, 4*(i-17):4*(i-16)] = output_data[location, i, 13]
            PSDs[1, 4*(i-17):4*(i-16)] = output_data[location, i, 14]
            PSDs[2, 4*(i-17):4*(i-16)] = output_data[location, i, 15]
            PSDs[3, 4*(i-17):4*(i-16)] = output_data[location, i, 16]
        for i in range(0,7):
            PSDs[0, 4*(i+7):4*(i+8)] = output_data[location, i, 13]
            PSDs[1, 4*(i+7):4*(i+8)] = output_data[location, i, 14]
            PSDs[2, 4*(i+7):4*(i+8)] = output_data[location, i, 15]
            PSDs[3, 4*(i+7):4*(i+8)] = output_data[location, i, 16]
        dwelling_indicators[location,0:4] = SDI(PSDs)
        
    # L50 to ARP
    for location in range(output_data.shape[0]):
        dwelling_indicators[location,4] = np.sum(np.maximum(0.0, output_data[location, 16:21, 1] - 50.0))/5.0
        
    # Lnight and Lden for reference
    for location in range(output_data.shape[0]):
        dwelling_indicators[location,5] = 10.0*np.log10((np.power(10.0,output_data[location,23,0]/10.0)+np.sum(np.power(10.0,output_data[location,0:7,0]/10.0)))/8.0)
        dwelling_indicators[location,6] = 10.0*np.log10( 
            np.sum(np.power(10.0,output_data[location,7:19,0]/10.0)/24) + 
            np.sum(np.power(10.0,(output_data[location,19:23,0]+5.0)/10.0)/24 + 
            (np.power(10.0,(output_data[location,23,0]+10.0)/10.0)+np.sum(np.power(10.0,(output_data[location,0:7,0]+10.0)/10.0)))/24.0))


    # then combine per dwelling
    dwelling_cols = ["SDI_0_3l", "SDI_3_6l", "SDI_6_12l", "SDI_12_18l", "ARP", "Lnightl", "SDI_0_3m", "SDI_3_6m", "SDI_6_12m", "SDI_12_18m", "Lnightm" ]
    dwelling_agg_indicators = np.zeros((output_data.shape[0],len(dwelling_cols)))
    Mpoints = 0
    agg_geom = []
    #agg_geom = np.zeros((output_data.shape[0],1))
    #agg_location = []
    agg_PGTPI = []
    #agg_PGTPI = np.zeros((output_data.shape[0],1))
    for ilocation in range(Npoints):
        location = PGTPI[ilocation]
        loc_where = np.where(output_geom_a[:,0] == location) 
        if len(loc_where[0])>0:
            #print(loc_where)
            dwelling_agg_indicators[Mpoints:Mpoints+1,0:4] = np.squeeze(np.amin(dwelling_indicators[loc_where[0],0:4],axis=0))
            dwelling_agg_indicators[Mpoints:Mpoints+1,4] = np.amin(dwelling_indicators[loc_where[0],4],axis=0)
            dwelling_agg_indicators[Mpoints:Mpoints+1,5] = np.amin(dwelling_indicators[loc_where[0],5],axis=0)
            dwelling_agg_indicators[Mpoints:Mpoints+1,6:10] = np.squeeze(np.amax(dwelling_indicators[loc_where[0],0:4],axis=0))
            dwelling_agg_indicators[Mpoints:Mpoints+1,10] = np.amax(dwelling_indicators[loc_where[0],5],axis=0)
            iwhere = PGTPI[PGTPI == location].index.to_list()
            if len(iwhere) == 1:
                #print(data[iwhere].item())
                agg_geom.append(data[iwhere].item())
                #agg_location.append(location)
                agg_PGTPI.append(location)
                #agg_PGTPI[Mpoints:Mpoints+1] = location
                Mpoints += 1
            else: 
                print('duplicated unique_identifier detected ' + str(location) + 'index ' + str(iwhere))
    
    # now save
    dictionairy = {dwelling_cols[j]: dwelling_agg_indicators[0:Mpoints, j] for j in range(len(dwelling_cols))}
    #dictionairy["dwelling"] = agg_location #we use unique PGTPI for dwelling
    dictionairy[PID] = agg_PGTPI
    dictionairy["geometry"] = agg_geom
    df_dwelling = DataFrame(dictionairy)
    #print(df_dwelling)
    #print(df_dwelling['geometry'].to_string())
    #df_dwelling['geometry'] = geopandas.GeoSeries.from_wkt(df_dwelling['geometry'].to_string())
    gdf_dwelling = GeoDataFrame(df_dwelling, geometry = 'geometry', crs=crs_of_receiver)
    #gdf_dwelling.set_geometry(col='geometry', inplace=True)
    gdf_dwelling.to_file(file_dwellingshape_save)

    print(list(gdf_dwelling.columns))
        
        

    # =============================================================================================
    # adding additional indicators
    hourly = gdf #geopandas.read_file(file_hourlyshape_save)
    print(list(hourly.columns))

    # 24 hour LAeq as a total exposure dose at home
    LAeqs = hourly.loc[:, 'LAeq@0h':'LAeq@23h']
    dwelling = hourly.get(['dwelling'])
    expLAeqs = np.power(10.0,LAeqs/10.0)
    expLAeq = np.sum(expLAeqs,axis=1)
    LAeq = 10.0*np.log10(expLAeq/24)
    hourly = hourly.drop('geometry',axis=1)
    hourly = hourly.assign(LAeq=LAeq)
    
    # LA90 evening
    LA90s_ev = hourly.loc[:, 'LA90@16h':'LA90@21h']
    dwelling = hourly.get(['dwelling'])
    LA90_ev = np.sum(LA90s_ev,axis=1)
    LA90_ev = LA90_ev/6
    #hourly = hourly.drop('geometry',axis=1)
    hourly = hourly.assign(LA90_ev=LA90_ev)
    
    
    # N90 evening
    N90s_ev = hourly.loc[:, 'N90@16h':'N90@21h']
    dwelling = hourly.get(['dwelling'])
    N90_ev = np.sum(N90s_ev,axis=1)
    N90_ev = N90_ev/6
    #hourly = hourly.drop('geometry',axis=1)
    hourly = hourly.assign(N90_ev=N90_ev)
    
    # LA50 evening
    LA50s_ev = hourly.loc[:, 'LA50@16h':'LA50@21h']
    dwelling = hourly.get(['dwelling'])
    LA50_ev = np.sum(LA50s_ev,axis=1)
    LA50_ev = LA50_ev/6
    #hourly = hourly.drop('geometry',axis=1)
    hourly = hourly.assign(LA50_ev=LA50_ev)
    
                
    # N50 evening
    N50s_ev = hourly.loc[:, 'N50@16h':'N50@21h']
    dwelling = hourly.get(['dwelling'])
    N50_ev = np.sum(N50s_ev,axis=1)
    N50_ev = N50_ev/6
    #hourly = hourly.drop('geometry',axis=1)
    hourly = hourly.assign(N50_ev=N50_ev)
    
    # LAeq evening
    LAeqs_ev = hourly.loc[:, 'LAeq@16h':'LAeq@21h']
    dwelling = hourly.get(['dwelling'])
    expLAeqs_ev = np.power(10.0,LAeqs_ev/10.0)
    expLAeq_ev = np.sum(expLAeqs_ev,axis=1)
    LAeq_ev = 10.0*np.log10(expLAeq_ev/6)
    #hourly = hourly.drop('geometry',axis=1)
    hourly = hourly.assign(LAeq_ev=LAeq_ev)
    
    # Lden child
    # 3 to 12 year old; school and kindergarten
    Le = hourly.loc[:, 'LAeq@16h':'LA50@20h']
    Ln1 = hourly.loc[:, 'LAeq@21h':'LA50@23h']
    Ln2 = hourly.loc[:, 'LAeq@0h':'LA50@7h']
    dwelling = hourly.get(['dwelling'])
    expLens = np.sum(np.power(10.0,(Le+5)/10.0),axis=1)/5 + np.sum(np.power(10.0,(Ln1+10)/10.0),axis=1)/3 + np.sum(np.power(10.0,(Ln2+10)/10.0),axis=1)/8
    Len_3_12 = 10.0*np.log10(expLens)
    hourly = hourly.assign(Len_3_12=Len_3_12)
    # 12 to 18 year old; school and kindergarten
    Le = hourly.loc[:, 'LAeq@17h':'LA50@21h']
    Ln1 = hourly.loc[:, 'LAeq@22h':'LA50@23h']
    Ln2 = hourly.loc[:, 'LAeq@0h':'LA50@7h']
    dwelling = hourly.get(['dwelling'])
    expLens = np.sum(np.power(10.0,(Le+5)/10.0),axis=1)/5 + np.sum(np.power(10.0,(Ln1+10)/10.0),axis=1)/2 + np.sum(np.power(10.0,(Ln2+10)/10.0),axis=1)/8
    Len_12_18 = 10.0*np.log10(expLens)
    hourly = hourly.assign(Len_12_18=Len_12_18)
    
    
    
    print(list(hourly.columns))
    
    # # emergence of peaks; notice model
    # EL50_10 = hourly.loc[:, 'ETLA50+116':'ETLA50+120']
    # EL50_3 = hourly.loc[:, 'ETLA50+3_7':'ETLA50+311' ]
    # alfa = (np.arange(5)+5)/5
    # EPE = np.sum(np.multiply(EL50_10,alfa),axis=1)+np.sum(np.multiply(EL50_3,alfa),axis=1)
    # hourly = hourly.assign(EPE=EPE)
    
    # # emergence of peaks; notice model
    # use originale col names   'ETLA50+10@19h' and 'ETLA50+3@17h'
    EL50_10 = hourly.loc[:, 'ETLA50+10@17h':'ETLA50+10@21h']
    EL50_3 = hourly.loc[:, 'ETLA50+3@17h':'ETLA50+3@21h' ]
    alfa = (np.arange(5)+5)/5
    EPE = np.sum(np.multiply(EL50_10,alfa),axis=1)+np.sum(np.multiply(EL50_3,alfa),axis=1)
    hourly = hourly.assign(EPE=EPE)
    
    dwellingkey =  PID   #'dwelling'  # 'adrkey'  #print(list(hourly.columns)) #PID
    #unique_index = PID   #'dwelling'  #'adrkey' #PID
    mostexposed = hourly.groupby(by=PID,as_index=False).agg('max')
    leastexposed = hourly.groupby(by=PID,as_index=False).agg('min')

    # now add the new indicators to the list of indicators per dwelling
    new_indicator_list = ['LAeq', 'LA90_ev', 'N90_ev', 'LA50_ev', 'N50_ev', 'LAeq_ev', 'Len_3_12', 'Len_12_18', 'EPE'] 
    Most = mostexposed.get([dwellingkey]+new_indicator_list)
    Least = leastexposed.get([dwellingkey]+new_indicator_list)
    Most[dwellingkey] = Most[dwellingkey].astype('int64')
    Least[dwellingkey] = Least[dwellingkey].astype('int64')
    
    new_indicator_list_mo = [s + 'm' for s in new_indicator_list]
    mapdict = dict(zip(new_indicator_list, new_indicator_list_mo))
    Most = Most.rename(columns = mapdict)
    
    new_indicator_list_le = [s + 'l' for s in new_indicator_list]
    mapdict = dict(zip(new_indicator_list, new_indicator_list_le))
    Least = Least.rename(columns = mapdict)
    
    gdf_dwelling = gdf_dwelling.merge(Most, left_on=dwellingkey, right_on=dwellingkey)
    gdf_dwelling = gdf_dwelling.merge(Least, left_on=dwellingkey, right_on=dwellingkey)
    gdf_dwelling.to_file(file_dwellingshape_save)
    gdf.to_file(file_hourlyshape_save)
