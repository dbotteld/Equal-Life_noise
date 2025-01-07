from geopandas import *
from shapely.geometry import box
from shapely.ops import transform
from shapely.geometry import Point, MultiPoint, shape
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
import hdbscan 
from hdbscan.flat import HDBSCAN_flat
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture as GMM
from typing import List, Set, Dict, Tuple

def create_grid(total_bounds: List[float], cell_size: float, crs: str) -> GeoDataFrame:
    [xmin, ymin, xmax, ymax] = total_bounds
    grid_cells = []
    for x0 in np.arange(xmin, xmax+cell_size, cell_size ):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            # bounds
            x1 = x0-cell_size
            y1 = y0+cell_size
            grid_cells.append(box(x0, y0, x1, y1))
    cell = GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
    return cell

def create_partition(gdf: GeoDataFrame, cell_size: float, crs) -> GeoDataFrame:
    cell = create_grid(gdf.to_crs(crs).total_bounds, cell_size, crs)

    if crs != gdf.crs: 
        print("cell transformed")
        cell = cell.to_crs(gdf.crs)

#================ plot grid ====================================================
#temporarily disabled due to poor implementation of Qt on waves03
    #ax = gdf.plot(markersize=.1, figsize=(12, 8), cmap='jet')
    # plt.autoscale(False)
    #cell.plot(ax=ax, facecolor="none", edgecolor='grey')
    #world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
    #print(gdf.crs)
    #world.to_crs(gdf.crs).plot(ax=ax, color='none', edgecolor='black')
    #plt.show()
#=====================================================================================

    merged = sjoin(gdf, cell, how='left', op='within')
    return merged

def create_clustered_partition(gdf: GeoDataFrame, n_clusters:int = 90, crs = 4326,  max_cluster_size:float = 150000, min_cluster_n = 5): 
    data = np.array([point.coords[0] for point in gdf.to_crs(crs).geometry])
    # clusterer = HDBSCAN_flat(data, n_clusters = 50, min_cluster_size = min_cluster_n, core_dist_n_jobs = 12)
    clusterer = KMeans(n_clusters=n_clusters)
    clusterer.fit(data)
    # clusterer = GMM(n_components = 100).fit(data)
    # labels = clusterer.predict(data)
    labels = clusterer.labels_
    # print(clusterer.labels_)

    clusters = pd.DataFrame({"x":data[:, 0], "y":data[:, 1], "cluster":labels})

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    groups = clusters.groupby('cluster')
    for name, group in groups:
        points = list(zip(group.x, group.y))
        mpt = MultiPoint([Point(point) for point in points])
        c_hull = mpt.convex_hull.buffer(1000)
        # c_hull = ConvexHull(list(zip(group.x, group.y)))
        # convex_hull_plot_2d(c_hull, ax=ax)
        #plt.plot(*c_hull.exterior.xy) # temporarily removed because of Qt poorly installed
        
        #ax.plot(group.x, group.y, marker='o', linestyle='', markersize=12, label=name)

    plt.legend()
    plt.show()

def create_clustered_partition2(gdf: GeoDataFrame, crs = 4326,  max_cluster_size:float = 40000**2, buffer = 2000): 
    print(gdf.size)
    data = np.array([point.coords[0] for point in gdf.to_crs(crs).geometry])
    # clusterer = HDBSCAN_flat(data, n_clusters = 50, min_cluster_size = min_cluster_n, core_dist_n_jobs = 12)
    clusterer = KMeans(n_clusters=2)
    clusterer.fit(data)
    # clusterer = GMM(n_components = 100).fit(data)
    # labels = clusterer.predict(data)
    labels = clusterer.labels_
    # print(clusterer.labels_)
    gdf["label"] = labels
    # clusters = pd.DataFrame({"x":data[:, 0], "y":data[:, 1], "cluster":labels})
    # gdf = geopandas.GeoDataFrame(clusters, geometry=geopandas.points_from_xy(clusters['x'], clusters['y']), crs=crs)

    groups = gdf.groupby('label')
    cluster_polys_buffer = []
    cluster_polys = []
    for name, group in groups:
        mpt = MultiPoint([point for point in group.to_crs(crs)['geometry']])
        c_hull = mpt.convex_hull
        c_hull_buffer = c_hull.buffer(buffer)
        c_hull = c_hull.buffer(5)
        if c_hull_buffer.area > max_cluster_size:
            new_cluster_polys, new_cluster_polys_buffer = create_clustered_partition2(group, crs=crs, max_cluster_size=max_cluster_size)
            cluster_polys.extend(new_cluster_polys)
            cluster_polys_buffer.extend(new_cluster_polys_buffer)
        else:
            cluster_polys.append(c_hull)
            cluster_polys_buffer.append(c_hull_buffer)

    return cluster_polys, cluster_polys_buffer

def create_part(gdf: GeoDataFrame, crs = 4326,  max_cluster_size:float = 40000**2, buffer = 2000):
    cluster_polys, cluster_polys_buffer = create_clustered_partition2(gdf, crs, max_cluster_size, buffer)
    #plot_clusters(cluster_polys_buffer, gdf, crs)
    cell = GeoDataFrame(cluster_polys, columns=['geometry'], crs=crs).to_crs(gdf.crs)
    cell_buffer = GeoDataFrame(cluster_polys_buffer, columns=['geometry'], crs=crs).to_crs(gdf.crs)
    
    merged = sjoin(gdf, cell, how='left', op='within')
    print(merged)
    return merged, cell_buffer


def plot_clusters(cluster_polys, gdf, crs = 4326):
    ax = gdf.to_crs(crs).plot(markersize=.1, figsize=(12, 8), cmap='jet')
    for poly in cluster_polys:
        ax.plot(*poly.exterior.xy)
    plt.show()

def create_bounding_box(bbox, buffer, point_in_Cartesian, projection_func, projection_func_inv): 
    [minx,miny,maxx,maxy] = bbox
    if point_in_Cartesian:
        southwest_cart = Point(minx-buffer, miny-buffer) 
        northeast_cart = Point(maxx+buffer, maxy+buffer)
        southwest_osm: Point = transform(projection_func_inv, southwest_cart)
        northeast_osm: Point = transform(projection_func_inv, northeast_cart)
    else:
        southwest_osm = Point(minx, miny) 
        northeast_osm = Point(maxx, maxy)
        southwest_cart: Point = transform(projection_func, southwest_osm)
        northeast_cart: Point = transform(projection_func, northeast_osm)
        southwest_cart = Point(southwest_cart.x-buffer, southwest_cart.y-buffer)
        northeast_cart = Point(northeast_cart.x+buffer, northeast_cart.y+buffer)
        southwest_osm: Point = transform(projection_func_inv, southwest_cart)
        northeast_osm: Point = transform(projection_func_inv, northeast_cart)
    return southwest_cart, northeast_cart, southwest_osm, northeast_osm 