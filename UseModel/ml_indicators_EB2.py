import pyproj
import math
import numpy as np
import osmnx as ox
import pandas as pd
import matplotlib.pyplot as plt
from geopandas import GeoDataFrame, GeoSeries
from shapely.wkt import loads
from shapely.geometry import Point, LineString, Polygon
from shapely.affinity import rotate
from shapely.ops import transform
from time import process_time

# TODO: create functions such as func(point, buildings, roads)
# TODO: functions list :
# DONE :        - first intersect distance per angle
# DONE :        - building density per angle
# DONE :        - object distance per category per angle
# DONE :        - add traffic intensity to roads
# DONE :        - length of roads based on traffic intensity, between min and max radius
# DONE :        - surface of buildings, between min and max radius
# TODO :        - surface of land use (green, industrial), between min and max radius


# FIXME : cleanup buidlings (small area, small height) for far radius

# scenario Flanders for ML (Colin)
aadf_d = [20400, 8400, 4800, 6600, 700, 350, 175, 0]
aadf_e = [20400, 1600, 1000, 1200, 200, 100, 50, 0]
aadf_n = [20400, 800, 640, 720, 100, 50, 25, 0]
#hv_d = [0.15, 0.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
#hv_e = [0.11, 0.11, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
#hv_n = [0.32, 0.32, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
hv_d = [0.15, 0.15, 0.03, 0.06, 0.02, 0.01, 0.01, 0.00]
hv_e = [0.11, 0.11, 0.03, 0.06, 0.02, 0.01, 0.01, 0.00]
hv_n = [0.32, 0.32, 0.03, 0.06, 0.02, 0.01, 0.01, 0.00]
#Edge Betweenness
lv_EB = [0, 0, 800, 600, 200, 70, 30, 0]
EBcross = [0, 0, 2E6, 1E6, 7E5, 1.5E5, 1E5, 0]
EBslope = [0, 0, 5.38E-7, 1.08E-6, 1.54E-6, 7.17E-6, 1.08E-5, 0]
#Diurnal pattern
Csat = [0.084, 0.055, 0.04, 0.04, 0.058, 0.122, 0.247, 0.359, 0.384, 0.372, 0.386, 0.397, 0.403, 0.409, 0.412, 0.432, 0.475, 0.495, 0.449, 0.338, 0.244, 0.191, 0.159, 0.12, 0.41] #last one if nodiurnal pattern
Fhv = [0.16, 0.24, 0.32, 0.4, 0.43, 0.39, 0.3, 0.19, 0.15, 0.18, 0.19, 0.18, 0.18, 0.17, 0.17, 0.16, 0.13, 0.1, 0.1, 0.11, 0.12, 0.12, 0.1, 0.12]
primaryocc = [0.77, 0.73, 0.45, 0.32, 0.27, 0.55, 0.95, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.95, 0.89, 0.8]
secondaryocc = [0.88, 0.85, 0.5, 0.35, 0.29, 0.62, 0.94, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.97, 0.94, 0.91]
tertiaryocc = [0.63, 0.56, 0.31, 0.25, 0.19, 0.19, 0.81, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.94, 0.81, 0.75]
residentialocc = [0.6, 0.5, 0.15, 0.13, 0.13, 0.15, 0.9, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.75, 0.6]
livingstreetocc = [0.56, 0.44, 0.19, 0.19, 0.13, 0.19, 0.88, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.75, 0.5]
primaryhv = [0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03]
secondaryhv = [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.05, 0.04]
tertiaryhv = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
residlivinghv = [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

speed = [130, 110, 80, 80, 50, 30, 30, 0]

#aadf_d = [26103, 17936, 7124, 1400, 700, 350, 175, 0]
#aadf_e = [7458, 3826, 1069, 400, 200, 100, 50, 0]
#aadf_n = [3729, 2152, 712, 200, 100, 50, 25, 0]
#hv_d = [0.25, 0.2, 0.2, 0.15, 0.10, 0.05, 0.02, 0.00]
#hv_e = [0.35, 0.2, 0.15, 0.10, 0.06, 0.02, 0.01, 0.00]
#hv_n = [0.45, 0.2, 0.1, 0.05, 0.03, 0.01, 0.0, 0.00]
#speed = [130, 110, 80, 80, 50, 30, 30, 0]

hours_in_d: int = 12
hours_in_e: int = 4
hours_in_n: int = 8


def get_road_category(highway) -> int:
    if highway in ["motorway", "motorway_link"]:
        return 0
    elif highway in ["trunk", "trunk_link"]:
        return 1
    elif highway in ["primary", "primary_link"]:
        return 2
    elif highway in ["secondary", "secondary_link"]:
        return 3
    elif highway in ["tertiary", "tertiary_link", "unclassified"]:
        return 4
    elif highway in ["residential"]:
        return 5
    elif highway in ["service", "living_street"]:
        return 6
    else:
        return 7


def get_nb_lv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between=False) -> int:       #Period can also be time of day         edge betweenness can also be float to accomodate for edge betweenness
    FEB: float = 1
    hour: int = 24
    timegiven = isinstance(period, int)
    if timegiven:
        hour = period
        period = 'd'
    if category == 0 or category == 1:
        if float(lanes) == 1 or math.isnan(float(lanes)):
            if typeofcat == "motorway" or typeofcat == "trunk":
                if period == "d":
                    #lv = int((1 - hv_d[category]) * aadf_d[category] / hours_in_d)
                    lv = 1700 * Csat[hour]
                elif period == "e":
                    #lv = int((1 - hv_e[category]) * aadf_e[category] / hours_in_e)
                    lv = 1700 * 0.18
                else:  #n
                    #lv = int((1 - hv_n[category]) * aadf_n[category] / hours_in_n)
                    lv = 1700 * 0.09
            else: #motorway_link or trunk_link
                lv = 1700 * 0.1
        else: #lanes>1
            if period =="d":
                lv = (2376 - 95.8936 * float(lanes)) * float(lanes) * Csat[hour]
            elif period =="e":
                lv = (2376 - 95.8936 * float(lanes)) * float(lanes) * 0.18
            else: #period =="n"
                lv = (2376 - 95.8936 * float(lanes)) * float(lanes) * 0.09
#    elif category == 1:      #trunk/trunk_link
#        if float(lanes) == 1 or math.isnan(float(lanes)):
#            if period == "d":
#                lv = aadf_d[category] / hours_in_d
#            elif period == "e":
#                lv = aadf_e[category] / hours_in_e
#            else: #n
#                lv = aadf_n[category] / hours_in_n
#        else: #more lanes
#            if period == "d":
#                lv = aadf_d[category] / hours_in_d * float(lanes)
#            elif period == "e":
#                lv = aadf_e[category] / hours_in_e * float(lanes)
#            else: #n
#                lv = aadf_n[category] / hours_in_n * float(lanes)
    elif isinstance(Edge_between, bool):
        if period == "d":
            lv = aadf_d[category] / hours_in_d
        elif period == "e":
            lv = aadf_e[category] / hours_in_e
        else:
            lv = aadf_n[category] / hours_in_n
    else:   #Edge betweenness
        lv = lv_EB[category]
        MinT = 0.01
        MaxT = 2.5
        EBc = EBcross[category]
        EBs = EBslope[category]
        FEB = max(MinT, min(MaxT, 1 + (Edge_between - EBc) * EBs))
    if timegiven and category > 1:
        if category == 2:
            lv *= primaryocc[hour]
        elif category == 3:
            lv *= secondaryocc[hour]
        elif category == 4:
            lv *= tertiaryocc[hour]
        elif category == 5:
            lv *= residentialocc[hour]
        else:
            lv *= livingstreetocc[hour]
    if oneway and category > 1: #oneway streets with multiple lanes get more traffic
        if math.isnan(float(lanes)) or int(lanes) == 1:
            lv /= 2
    if tunnel == 'yes':
        lv = 0
    return round(lv * FEB)


def get_nb_hv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between=False) -> int:
    timegiven = isinstance(period, int)
    if timegiven:
        hour = period
    if not timegiven:
        if period == "d":
            hv = hv_d[category] * get_nb_lv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between)
            #hv = int(hv_d[category] * aadf_d[category] / hours_in_d)
        elif period == "e":
            hv = hv_e[category] * get_nb_lv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between)
            #hv = int(hv_e[category] * aadf_e[category] / hours_in_e)
        else:  # n
            hv = hv_n[category] * get_nb_lv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between)
            #hv = int(hv_n[category] * aadf_n[category] / hours_in_n)
    elif category < 2:
        hv = Fhv[hour] * get_nb_lv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between)
    elif category == 2:
        hv = primaryhv[hour] * get_nb_lv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between)
    elif category == 3:
        hv = secondaryhv[hour] * get_nb_lv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between)
    elif category == 4:
        hv = tertiaryhv[hour] * get_nb_lv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between)
    else:
        hv = residlivinghv[hour] * get_nb_lv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between)
    return round(hv)


#def get_nb_hv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between=False) -> int:
#    lv: int = 0
#    if period == "d":
#        hv = int(hv_d[category] * get_nb_lv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between))
#        #hv = int(hv_d[category] * aadf_d[category] / hours_in_d)
#    elif period == "e":
#        hv = int(hv_e[category] * get_nb_lv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between))
#        #hv = int(hv_e[category] * aadf_e[category] / hours_in_e)
#    else:  # n
#        hv = int(hv_n[category] * get_nb_lv(period, category, oneway, tunnel, lanes, typeofcat, Edge_between))
#        #hv = int(hv_n[category] * aadf_n[category] / hours_in_n)
#    #if oneway:
#        #hv /= 2
#    return hv


def add_traffic_intensity(roads: GeoDataFrame) -> GeoDataFrame:
    roads["category"] = roads.apply(lambda road: get_road_category(road["highway"]), axis=1)
    if 'edge_betweenness' in roads.columns:
        roads["LV_N0"] = roads.apply(lambda road: get_nb_lv(0, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N1"] = roads.apply(lambda road: get_nb_lv(1, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N2"] = roads.apply(lambda road: get_nb_lv(2, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N3"] = roads.apply(lambda road: get_nb_lv(3, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N4"] = roads.apply(lambda road: get_nb_lv(4, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N5"] = roads.apply(lambda road: get_nb_lv(5, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N6"] = roads.apply(lambda road: get_nb_lv(6, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N7"] = roads.apply(lambda road: get_nb_lv(7, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N8"] = roads.apply(lambda road: get_nb_lv(8, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N9"] = roads.apply(lambda road: get_nb_lv(9, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N10"] = roads.apply(lambda road: get_nb_lv(10, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N11"] = roads.apply(lambda road: get_nb_lv(11, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N12"] = roads.apply(lambda road: get_nb_lv(12, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N13"] = roads.apply(lambda road: get_nb_lv(13, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N14"] = roads.apply(lambda road: get_nb_lv(14, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N15"] = roads.apply(lambda road: get_nb_lv(15, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N16"] = roads.apply(lambda road: get_nb_lv(16, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N17"] = roads.apply(lambda road: get_nb_lv(17, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N18"] = roads.apply(lambda road: get_nb_lv(18, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N19"] = roads.apply(lambda road: get_nb_lv(19, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N20"] = roads.apply(lambda road: get_nb_lv(20, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N21"] = roads.apply(lambda road: get_nb_lv(21, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N22"] = roads.apply(lambda road: get_nb_lv(22, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"], road["edge_betweenness"]), axis=1)
        roads["LV_N23"] = roads.apply(lambda road: get_nb_lv(23, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"], road["edge_betweenness"]), axis=1)
    else:
        roads["LV_N0"] = roads.apply(lambda road: get_nb_lv(0, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N1"] = roads.apply(lambda road: get_nb_lv(1, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N2"] = roads.apply(lambda road: get_nb_lv(2, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N3"] = roads.apply(lambda road: get_nb_lv(3, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N4"] = roads.apply(lambda road: get_nb_lv(4, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N5"] = roads.apply(lambda road: get_nb_lv(5, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N6"] = roads.apply(lambda road: get_nb_lv(6, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N7"] = roads.apply(lambda road: get_nb_lv(7, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N8"] = roads.apply(lambda road: get_nb_lv(8, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N9"] = roads.apply(lambda road: get_nb_lv(9, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N10"] = roads.apply(lambda road: get_nb_lv(10, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N11"] = roads.apply(lambda road: get_nb_lv(11, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N12"] = roads.apply(lambda road: get_nb_lv(12, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"]), axis=1)
        roads["LV_N13"] = roads.apply(lambda road: get_nb_lv(13, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"]), axis=1)
        roads["LV_N14"] = roads.apply(lambda road: get_nb_lv(14, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"]), axis=1)
        roads["LV_N15"] = roads.apply(lambda road: get_nb_lv(15, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"]), axis=1)
        roads["LV_N16"] = roads.apply(lambda road: get_nb_lv(16, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"]), axis=1)
        roads["LV_N17"] = roads.apply(lambda road: get_nb_lv(17, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"]), axis=1)
        roads["LV_N18"] = roads.apply(lambda road: get_nb_lv(18, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"]), axis=1)
        roads["LV_N19"] = roads.apply(lambda road: get_nb_lv(19, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"]), axis=1)
        roads["LV_N20"] = roads.apply(lambda road: get_nb_lv(20, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N21"] = roads.apply(lambda road: get_nb_lv(21, road["category"], road["oneway"], road["tunnel"], road["lanes"], road["highway"]), axis=1)
        roads["LV_N22"] = roads.apply(lambda road: get_nb_lv(22, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"]), axis=1)
        roads["LV_N23"] = roads.apply(lambda road: get_nb_lv(23, road["category"], road["oneway"], road["tunnel"], road["lanes"],road["highway"]), axis=1)
    return roads.copy()

def get_traffic_intensity(roads: GeoDataFrame, index) -> GeoDataFrame:
    category = get_road_category(roads["highway"][index])
    if 'edge_betweenness' in roads.columns:
        LV_N00 = get_nb_lv(0, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N01 = get_nb_lv(1, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N02 = get_nb_lv(2, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N03 = get_nb_lv(3, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N04 = get_nb_lv(4, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N05 = get_nb_lv(5, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N06 = get_nb_lv(6, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N07 = get_nb_lv(7, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N08 = get_nb_lv(8, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N09 = get_nb_lv(9, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N10 = get_nb_lv(10, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N11 = get_nb_lv(11, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N12 = get_nb_lv(12, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N13 = get_nb_lv(13, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N14 = get_nb_lv(14, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N15 = get_nb_lv(15, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N16 = get_nb_lv(16, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N17 = get_nb_lv(17, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N18 = get_nb_lv(18, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N19 = get_nb_lv(19, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N20 = get_nb_lv(20, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N21 = get_nb_lv(21, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N22 = get_nb_lv(22, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
        LV_N23 = get_nb_lv(23, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index], roads["edge_betweenness"][index])
    else:
        LV_N00 = get_nb_lv(0, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N01 = get_nb_lv(1, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N02 = get_nb_lv(2, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N03 = get_nb_lv(3, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N04 = get_nb_lv(4, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N05 = get_nb_lv(5, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N06 = get_nb_lv(6, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N07 = get_nb_lv(7, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N08 = get_nb_lv(8, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N09 = get_nb_lv(9, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N10 = get_nb_lv(10, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N11 = get_nb_lv(11, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N12 = get_nb_lv(12, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N13 = get_nb_lv(13, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N14 = get_nb_lv(14, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N15 = get_nb_lv(15, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N16 = get_nb_lv(16, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N17 = get_nb_lv(17, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N18 = get_nb_lv(18, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N19 = get_nb_lv(19, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N20 = get_nb_lv(20, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N21 = get_nb_lv(21, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N22 = get_nb_lv(22, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
        LV_N23 = get_nb_lv(23, category, roads["oneway"][index], roads["tunnel"][index], roads["lanes"][index], roads["highway"][index])
    return np.array([LV_N00, LV_N01, LV_N02, LV_N03, LV_N04, LV_N05, LV_N06, LV_N07, LV_N08, LV_N09, LV_N10, LV_N11, LV_N12, LV_N13, LV_N14, LV_N15, LV_N16, LV_N17, LV_N18, LV_N19, LV_N20, LV_N21, LV_N22, LV_N23])


def get_closest_object_distance_by_angle(point: Point, distance: int, nb_angles: int, objects: GeoDataFrame) -> dict:
    """
    get the closest object "seen" from a point at different angles

    :param point: the center point
    :param distance: the max distance from point to consider
    :param nb_angles: the number of angles. The angle precision will be 360/nb_angles
    :param objects: a GeoDataFrame containing buildings or roads
    :return: dict[angle, distance].
        The distance is the minimum intersect distance found in the objects for that angle (-1 if nothing is found)
    """
    result: dict = {}
    end = Point(point.x + distance, point.y)
    horizontal_line = LineString([point, end])
    rad_angle = math.radians(-360 // nb_angles)
    prev_line = rotate(horizontal_line, rad_angle, origin=point, use_radians=True)

    for angle in range(0, 360, 360 // nb_angles):
        rad_angle = math.radians(angle)
        line = rotate(horizontal_line, rad_angle, origin=point, use_radians=True)
        polygon = Polygon([point, prev_line.coords[1], line.coords[1]])
        prev_line = line

        closest_distance = -2
        intersect_objects = objects.copy().loc[objects.intersects(polygon)]
        linepieces_intersect = intersect_objects.intersection(polygon)
        distlinepiece = []
        if len(intersect_objects) > 0:
            distlinepiece.append(linepieces_intersect.distance(point))
            closest_distance = min(distlinepiece)
            closest_distance = closest_distance.min()
            # intersect_objects["distance"] = intersect_objects["geometry"].distance(point)
            # closest_distance = intersect_objects["distance"].min()

        result[angle] = closest_distance

    return result


def get_building_surface_between_distances(point: Point, min_distance: int, max_distance: int, buildings: GeoDataFrame) -> float:
    """
    get total buidling surface between a min distance and a max distance from center point

    :param point: the center point
    :param min_distance: the min distance from point to consider
    :param max_distance: the max distance from point to consider
    :param buildings: a GeoDataFrame containing buildings
    :return: float.
        The total building surface between the two distances
    """

    result: float = 0.0

    close_circle = point.buffer(min_distance)
    large_circle = point.buffer(max_distance)

    polygon = large_circle - close_circle

    intersection = buildings.intersection(polygon)
    result = intersection.area.sum()

    return result

def get_building_surface_between_distances_fast(point: Point, min_distance: int, max_distance: int, buildings: GeoDataFrame) -> float:
    """
    get total buidling surface between a min distance and a max distance from center point.
    Fast version does not cut buildings outside of min and max distances

    :param point: the center point
    :param min_distance: the min distance from point to consider
    :param max_distance: the max distance from point to consider
    :param buildings: a GeoDataFrame containing buildings
    :return: float.
        The total building surface between the two distances
    """

    result: float = 0.0

    close_circle = point.buffer(min_distance)
    large_circle = point.buffer(max_distance)

    polygon = large_circle - close_circle

    result = buildings.loc[buildings.intersects(polygon)].area.sum()

    return result


def get_road_length_between_distances(point: Point, min_distance: int, max_distance: int, roads: GeoDataFrame) -> float:
    """
    get total road length between a min distance and a max distance from center point

    :param point: the center point
    :param min_distance: the min distance from point to consider
    :param max_distance: the max distance from point to consider
    :param buildings: a GeoDataFrame containing buildings
    :return: float.
        The total building surface between the two distances
    """

    result: float = 0.0

    close_circle = point.buffer(min_distance)
    large_circle = point.buffer(max_distance)

    polygon = large_circle - close_circle

    intersection = roads.intersection(polygon)
    #intersection = roads.intersects(polygon)
    result = intersection.length.sum()

    return result


def get_road_length_between_distances_fast(point: Point, min_distance: int, max_distance: int, roads: GeoDataFrame) -> float:
    """
    get total road length between a min distance and a max distance from center point
    Fast version does not cut roads outside of min and max distances

    :param point: the center point
    :param min_distance: the min distance from point to consider
    :param max_distance: the max distance from point to consider
    :param buildings: a GeoDataFrame containing buildings
    :return: float.
        The total building surface between the two distances
    """

    #result: float = 0.0

    close_circle = point.buffer(min_distance)
    large_circle = point.buffer(max_distance)

    polygon = large_circle - close_circle

    result = roads.loc[roads.intersects(polygon)].length.sum()

    return result

def get_building_surface_by_angle(point: Point, distance: int, nb_angles: int, buildings: GeoDataFrame) -> dict:
    """
    get total buidling density for every angle. every angle defines a polygon with the previous angle.

    :param point: the center point
    :param distance: the max distance from point to consider
    :param nb_angles: the number of angles. The angle precision will be 360/nb_angles
    :param buildings: a GeoDataFrame containing buildings
    :return: dict[angle, density].
        The density is the ratio of buildings surface divided by the angle polygon surface
    """

    result: dict = {}
    end = Point(point.x + distance, point.y)
    horizontal_line = LineString([point, end])
    rad_angle = math.radians(-360 // nb_angles)
    prev_line = rotate(horizontal_line, rad_angle, origin=point, use_radians=True)

    for angle in range(0, 360, 360 // nb_angles):
        rad_angle = math.radians(angle)
        line = rotate(horizontal_line, rad_angle, origin=point, use_radians=True)
        polygon = Polygon([point, prev_line.coords[1], line.coords[1]])
        intersection: GeoSeries = buildings.intersection(polygon)
        result[angle] = intersection.area.sum() / polygon.area
        prev_line = line

    return result


def get_first_object_by_angle(point: Point, distance: int, nb_angles: int, buildings: GeoDataFrame, roads: GeoDataFrame) -> dict:
    """
    get the first object "seen" from a point at different angles

    :param point: the center point
    :param distance: the max distance from point to consider
    :param nb_angles: the number of angles. The angle precision will be 360/nb_angles
    :param buildings: a GeoDataFrame containing buildings
    :param roads: a GeoDataFrame containing roads
    :return: dict[angle, tuple].
        The tuple is (object_type, distance, string) with object type an int such as
            0: nothing
            1: road
            2: building
    """
    result: dict = {}

    end = Point(point.x + distance, point.y)
    line = LineString([point, end])

    for angle in range(0, 360, 360 // nb_angles):
        rad_angle = math.radians(angle)
        ray = rotate(line, rad_angle, origin=point, use_radians=True)
        # print("Angle %d ; Ray geom : %s" % (angle, ray))

        closest_building_distance = 0
        df_ray_buildings = df_buildings.copy().loc[df_buildings.intersects(ray)]
        if len(df_ray_buildings) > 0:
            df_ray_buildings["ray"] = df_ray_buildings["geometry"].intersection(ray)
            df_ray_buildings["distance"] = df_ray_buildings["ray"].distance(point)
            closest_building = df_ray_buildings.loc[df_ray_buildings["distance"].idxmin()]
            closest_building_distance = closest_building["distance"]
            # print(closest_building)

        closest_road_distance = 0
        df_ray_roads = df_roads.copy().loc[df_roads.intersects(ray)]
        if len(df_ray_roads) > 0:
            df_ray_roads["ray"] = df_ray_roads["geometry"].intersection(ray)
            df_ray_roads["distance"] = df_ray_roads["ray"].distance(point)
            closest_road = df_ray_roads.loc[df_ray_roads["distance"].idxmin()]
            closest_road_distance = closest_road["distance"]
            # print(closest_road)

        if closest_building_distance == 0 and closest_road_distance == 0:
            result[angle] = (0, 0.0, "null")
            # print("Angle %d ; NOTHING" % angle)

        if closest_building_distance == 0 and not closest_road_distance == 0:
            result[angle] = (1, closest_road_distance, closest_road["highway"])
            # print("Angle %d ; ROAD (id: %s, type:%s); distance=%f" % (angle, closest_road.name[1], closest_road["highway"], closest_road_distance))

        if not closest_building_distance == 0 and closest_road_distance == 0:
            result[angle] = (2, closest_building_distance, "building")
            # print("Angle %d ; BUILDING (%s); distance=%f" % (angle, closest_building.name[1], closest_building_distance))

        if not closest_building_distance == 0 and not closest_road_distance == 0:
            if closest_building_distance < closest_road_distance:
                result[angle] = (2, closest_building_distance, "building")
                # print("Angle %d ; BUILDING (%s); distance=%f" % (angle, closest_building.name[1], closest_building_distance))
            else:
                result[angle] = (1, closest_road_distance, closest_road["highway"])
                # print("Angle %d ; ROAD (id: %s, type:%s); distance=%f" % (angle, closest_road.name[1], closest_road["highway"], closest_road_distance))

    return result


if __name__ == "__main__":

    start_time = process_time()

    # wkt_geom	CLUSTER	AREA_TYPE	HEIGHT	CLOSE_R	FAR_R	WIND	DELTA_LP
    # Point (3.72833799353678419 51.05331945170444641)	3	URBAN	4	500	2000	0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5	4

    # wkt_geom	CLUSTER	AREA_TYPE	HEIGHT	CLOSE_R	FAR_R	WIND	DELTA_LP
    # Point (3.72744962660784207 51.03180200357235918)	1	URBAN	4	500	2000	0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5	4

    receiver_point = Point(3.72744962660784207, 51.03180200357235918)
    length = 3200
    srid = 31370

    close_distance = 200
    nb_angles = 36

    df_buildings = ox.geometries_from_point((receiver_point.y, receiver_point.x), dist=length, tags={"building": True})
    df_roads = ox.geometries_from_point((receiver_point.y, receiver_point.x), dist=length, tags={"highway": True})

    loading_geom_time = process_time() - start_time
    print("Loading geometries : %f seconds" % loading_geom_time)

    temp_start_time = process_time()

    wgs84 = pyproj.CRS('EPSG:4326')
    target = pyproj.CRS("EPSG:%d" % srid)

    project = pyproj.Transformer.from_crs(wgs84, target, always_xy=True).transform

    df_buildings = df_buildings.to_crs("EPSG:%d" % srid)
    df_roads = df_roads.to_crs("EPSG:%d" % srid)

    point: Point = transform(project, receiver_point)

    print("Projecting geometries : %f seconds" % (process_time() - temp_start_time))

    temp_start_time = process_time()
    for i in range(len(df_roads)):
        traffic = get_traffic_intensity(df_roads, i)
    print("get_traffic_intensity_time : %f seconds" % (process_time() - temp_start_time))

    temp_start_time = process_time()
    df_roads = add_traffic_intensity(df_roads)
    print("add_traffic_intensity_time : %f seconds" % (process_time() - temp_start_time))

    temp_start_time = process_time()
    polygon = point.buffer(close_distance)
    df_close_roads = df_roads.loc[df_roads.intersects(polygon)]
    df_close_buildings = df_buildings.loc[df_buildings.intersects(polygon)]

    print(get_first_object_by_angle(point, close_distance, nb_angles, df_close_buildings, df_close_roads))
    print("get_first_object_by_angle_time : %f seconds" % (process_time() - temp_start_time))

    temp_start_time = process_time()
    print(get_building_surface_by_angle(point, close_distance, nb_angles, df_close_buildings))
    print("get_building_surface_by_angle_time : %f seconds" % (process_time() - temp_start_time))

    temp_start_time = process_time()
    print(get_closest_object_distance_by_angle(point, close_distance, nb_angles, df_close_buildings))
    print("get_closest_object_distance_by_angle : %f seconds" % (process_time() - temp_start_time))

    min_lv = 0
    for max_lv in [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 99999]:
        filtered_lv_roads = df_close_roads.loc[(df_close_roads["LV_N14"] > min_lv) & (df_close_roads["LV_N14"] <= max_lv)]
        if len(filtered_lv_roads) == 0:
            continue
        temp_start_time = process_time()
        print(get_closest_object_distance_by_angle(point, close_distance, nb_angles, filtered_lv_roads))
        print("get_closest_object_distance_by_angle (min_lv : %s ; max_lv : %s) : %f seconds" % (min_lv, max_lv, process_time() - temp_start_time))
        min_lv = max_lv

    min_lv = 200
    min_dist = 0
    for max_dist in [200, 400, 800, 1600, 3200]:
        temp_start_time = process_time()
        print(get_building_surface_between_distances(point, min_dist, max_dist, df_buildings))
        print("get_building_surface_between_distances_time (min_dist : %s ; max_dist : %s) : %f seconds" % (min_dist, max_dist, process_time() - temp_start_time))
        for max_lv in [400, 800, 1600, 99999]:
            filtered_lv_roads = df_roads.loc[(df_roads["LV_N14"] >= min_lv) & (df_roads["LV_N14"] < max_lv)]
            if len(filtered_lv_roads) == 0:
                continue
            temp_start_time = process_time()
            print(get_road_length_between_distances(point, min_dist, max_dist, filtered_lv_roads))
            print("get_road_length_between_distances_time (min_dist : %s ; max_dist : %s ; min_lv : %s ; max_lv : %s) : %f seconds" % (min_dist, max_dist, min_lv, max_lv, process_time() - temp_start_time))
            min_lv = max_lv
        min_dist = max_dist

    print("Total elapsed time : %f seconds" % (process_time() - start_time))