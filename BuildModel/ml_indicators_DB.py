import pyproj
import math
import osmnx as ox
from geopandas import GeoDataFrame, GeoSeries
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

# traffic in different scenario 
aadf_d = [[20400, 8400, 4800, 6600, 700, 350, 175, 0]]
aadf_e = [[20400, 1600, 1000, 1200, 200, 100, 50, 0]]
aadf_n = [[20400, 800, 640, 720, 100, 50, 25, 0]]
hv_d = [[0.15, 0.15, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]]
hv_e = [[0.11, 0.11, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]]
hv_n = [[0.32, 0.32, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]]
speed = [[130, 110, 80, 80, 50, 30, 30, 0]]

aadf_d.append( [20400, 4200, 2400, 2200, 200, 100, 40, 0])
aadf_e.append( [20400, 1600, 1000, 1200, 200, 100, 50, 0])
aadf_n.append( [20400, 800, 640, 720, 100, 50, 25, 0])
hv_d.append( [0.15, 0.15, 0.02, 0.01, 0.01, 0.00, 0.00, 0.00])
hv_e.append( [0.11, 0.11, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
hv_n.append( [0.32, 0.32, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00])
speed.append( [130, 110, 80, 80, 50, 30, 30, 0])


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


def get_nb_lv(period, category, oneway, lanes, typeofcat, scenario) -> int:
    lv: int = 0
    if category == 0:
        if float(lanes) == 1 or math.isnan(float(lanes)):
            if typeofcat == "motorway":
                if period == "d":
                    #lv = int((1 - hv_d[category]) * aadf_d[category] / hours_in_d)
                    lv = int(1700 * 0.41)
                elif period == "e":
                    #lv = int((1 - hv_e[category]) * aadf_e[category] / hours_in_e)
                    lv = int(1700 * 0.09)
                else:  #n
                    #lv = int((1 - hv_n[category]) * aadf_n[category] / hours_in_n)
                    lv = int(1700 * 0.1)
            else: #motorway_link
                lv = 1700 * 0.1
        else: #lanes>1
            if period =="d":
                lv = int((2376 - 95.8936 * float(lanes)) * float(lanes) * 0.41)
            elif period =="e":
                lv = int((2376 - 95.8936 * float(lanes)) * float(lanes) * 0.18)
            else: #period =="n"
                lv = int((2376 - 95.8936 * float(lanes)) * float(lanes) * 0.09)
    elif category == 1:      #trunk/trunk_link
        if float(lanes) == 1 or math.isnan(float(lanes)):
            if period == "d":
                lv = int(aadf_d[scenario-1][category] / hours_in_d)
            elif period == "e":
                lv = int(aadf_e[scenario-1][category] / hours_in_e)
            else: #n
                lv = int(aadf_n[scenario-1][category] / hours_in_n)
        else: #more lanes
            if period == "d":
                lv = int(aadf_d[scenario-1][category] / hours_in_d * float(lanes))
            elif period == "e":
                lv = int(aadf_e[scenario-1][category] / hours_in_e * float(lanes))
            else: #n
                lv = int(aadf_n[scenario-1][category] / hours_in_n * float(lanes))
    else: 
        if period == "d":
            lv = int(aadf_d[scenario-1][category] / hours_in_d)
        elif period == "e":
            lv = int(aadf_e[scenario-1][category] / hours_in_e)
        else:
            lv = int(aadf_n[scenario-1][category] / hours_in_n)
    if oneway and category > 1: #oneway streets with multiple lanes get more traffic
        if math.isnan(float(lanes)) or float(lanes) == 1:
            lv /= 2
    return lv


def get_nb_hv(period, category, oneway, lanes, typeofcat, scenario) -> int:
    lv: int = 0
    if period == "d":
        hv = int(hv_d[scenario-1][category] * get_nb_lv(period, category, oneway, lanes, typeofcat, scenario))
        #hv = int(hv_d[category] * aadf_d[category] / hours_in_d)
    elif period == "e":
        hv = int(hv_e[scenario-1][category] * get_nb_lv(period, category, oneway, lanes, typeofcat, scenario))
        #hv = int(hv_e[category] * aadf_e[category] / hours_in_e)
    else:  # n
        hv = int(hv_n[scenario-1][category] * get_nb_lv(period, category, oneway, lanes, typeofcat, scenario))
        #hv = int(hv_n[category] * aadf_n[category] / hours_in_n)
    #if oneway:
        #hv /= 2
    return hv


def add_traffic_intensity(roads: GeoDataFrame, scenario) -> GeoDataFrame:
	
    roads["category"] = roads.apply(lambda road: get_road_category(road["highway"]), axis=1)
    roads["lv_d"] = roads.apply(lambda road: get_nb_lv("d", road["category"], road["oneway"], road["lanes"], road["highway"], scenario), axis=1)
    roads["lv_e"] = roads.apply(lambda road: get_nb_lv("d", road["category"], road["oneway"], road["lanes"], road["highway"], scenario), axis=1)
    roads["lv_n"] = roads.apply(lambda road: get_nb_lv("d", road["category"], road["oneway"], road["lanes"], road["highway"], scenario), axis=1)
    roads["hv_d"] = roads.apply(lambda road: get_nb_hv("d", road["category"], road["oneway"], road["lanes"], road["highway"], scenario), axis=1)
    roads["hv_e"] = roads.apply(lambda road: get_nb_hv("d", road["category"], road["oneway"], road["lanes"], road["highway"], scenario), axis=1)
    roads["hv_n"] = roads.apply(lambda road: get_nb_hv("d", road["category"], road["oneway"], road["lanes"], road["highway"], scenario), axis=1)
    return roads.copy()


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
            #intersect_objects["distance"] = intersect_objects["geometry"].distance(point)
            #closest_distance = intersect_objects["distance"].min()

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
        filtered_lv_roads = df_close_roads.loc[(df_close_roads["lv_d"] > min_lv) & (df_close_roads["lv_d"] <= max_lv)]
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
            filtered_lv_roads = df_roads.loc[(df_roads["lv_d"] >= min_lv) & (df_roads["lv_d"] < max_lv)]
            if len(filtered_lv_roads) == 0:
                continue
            temp_start_time = process_time()
            print(get_road_length_between_distances(point, min_dist, max_dist, filtered_lv_roads))
            print("get_road_length_between_distances_time (min_dist : %s ; max_dist : %s ; min_lv : %s ; max_lv : %s) : %f seconds" % (min_dist, max_dist, min_lv, max_lv, process_time() - temp_start_time))
            min_lv = max_lv
        min_dist = max_dist

    print("Total elapsed time : %f seconds" % (process_time() - start_time))
