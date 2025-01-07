import networkx as nx
import osmnx as ox
import pandas as pd
import geopandas as gpd
import igraph as ig
import numpy as np
import os
from shapely.geometry import Point, LineString

import betweenness as EB

required_keys = ["oneway", 'tunnel', 'lanes', 'highway', "geometry"]


#####################################################################################
# METRICS #####################################################################################
#####################################################################################
def compute_node_closeness(G_nx, weight='length'):
    # create networkx graph
    osmids = list(G_nx.nodes)
    G_nx = nx.relabel.convert_node_labels_to_integers(G_nx)

    # give each node its original osmid as attribute since we relabeled them
    osmid_values = {k:v for k, v in zip(G_nx.nodes, osmids)}
    nx.set_node_attributes(G_nx, osmid_values, 'osmid')

    print("Convert to igraph ...")
    # convert networkx graph to igraph
    G_ig = ig.Graph(directed=True)
    G_ig.add_vertices(G_nx.nodes)
    G_ig.add_edges(G_nx.edges())
    G_ig.vs['osmid'] = osmids
    G_ig.es[weight] = list(nx.get_edge_attributes(G_nx, weight).values())
    assert len(G_nx.nodes()) == G_ig.vcount()
    assert len(G_nx.edges()) == G_ig.ecount()
    
    print("Calculating closeness ...")
    closeness1 = G_ig.closeness(vertices=None, mode='ALL', cutoff=None, weights=weight, normalized=True)
    
    print("Convert to dataframe closeness ...")
    gdf_nodes = ox.utils_graph.graph_to_gdfs(G_nx, nodes=True, edges=False, node_geometry=True, fill_edge_geometry=False)
    df_nodes = pd.DataFrame({'osmid': G_ig.vs["osmid"], 'node_closeness':closeness1})
    gdf_nodes = gdf_nodes.reset_index(drop=True)
    gdf_res = pd.merge(gdf_nodes, df_nodes, left_on='osmid', right_on='osmid', how='left')

    return gdf_res
    
    
def compute_edge_betweenness(G_nx, weight = 'length'):
    # create networkx graph
    osmids = list(G_nx.edges)
    G_nx = nx.relabel.convert_node_labels_to_integers(G_nx)
    osmid_values = {k:v for k, v in zip(G_nx.edges, osmids)}

    # # give each node its original osmid as attribute since we relabeled them
    nx.set_edge_attributes(G_nx, osmid_values, 'osmid')
    # print("Convert to igraph ...")
    # # convert networkx graph to igraph
    # G_ig = ig.Graph(directed=True)
    # G_ig.add_vertices(G_nx.nodes)
    # G_ig.add_edges(G_nx.edges())
    # G_ig.es['osmid'] = osmids
    # G_ig.es[weight] = list(nx.get_edge_attributes(G_nx, weight).values())
    # assert len(G_nx.nodes()) == G_ig.vcount()
    # assert len(G_nx.edges()) == G_ig.ecount()
    ###############################################
    # Calculating betweenness ###############################################
    ###############################################
    print("Calculating betweenness ...")
    # betweenness = G_ig.edge_betweenness(directed=True, cutoff=None, weights=weight)
    betweenness = EB.betweenness_centrality_parallel(G_nx, weight)
    print("Convert to dataframe betweenness ...")
    gdf_edges = ox.utils_graph.graph_to_gdfs(G_nx, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
    df_edges = pd.DataFrame(list(zip(osmid_values.values(), betweenness.values())), columns = ["osmid", "betweenness"])
    gdf_edges = gdf_edges.reset_index(drop=True)
    gdf_res = pd.merge(gdf_edges, df_edges, left_on='osmid', right_on='osmid', how='left')
    return gdf_res


def compute_edge_sinuosity(row):
    x, y = row.geometry.coords.xy
    start_pt = Point(x[0], y[0])
    end_pt = Point(x[-1], y[-1])
    straight_line = LineString((start_pt, end_pt))
    if straight_line.length:
        return row.geometry.length / straight_line.length
    else:
        return None

def get_streets_per_cities(cities, buffer_dist=0, network_type='drive', output_folder='.', 
                          intersection_clos=False, street_betw=False, street_sin=False, retain_all=True):
    #     drive: get drivable public streets (but not service roads)
    # drive_service: get drivable public streets including service roads
    # walk: get all streets and paths that pedestrians can use (this network type ignores one-way
    # directionality by always connecting adjacent nodes with reciprocal directed edges)
    # bike: get all streets and paths that cyclists can use
    # all: download all (non-private) OpenStreetMap streets and paths
    # all_private: download all OpenStreetMap streets and paths, including private-access 
    for city in cities:
        place_name = city  # Zwijndrecht, Gent

        G = ox.graph_from_place(place_name, network_type=network_type, buffer_dist=buffer_dist, retain_all=retain_all)
        place = ox.geocode_to_gdf(place_name)

        # store the street network if no enrichment is needed
        if not intersection_clos and not street_betw and not street_sin:
            edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            return edges

        # if Intersections - Nodes store closeness per intersection
        if intersection_clos:
            nodes = compute_node_closeness(G)
            nodes_clipped = gpd.clip(nodes, place)
            print("nodes done")
        
        # Streets - Edges
        if street_betw:
            edges = compute_edge_betweenness(G)
            print("edges betweenness done")
        
        if street_betw and street_sin:
            edges['edge_sinuosity'] = edges.apply(lambda row: compute_edge_sinuosity(row), axis=1)

        if street_sin and not street_betw:
            edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            edges['edge_sinuosity'] = edges.apply(lambda row: compute_edge_sinuosity(row), axis=1)
        
        #edge closeness centrality: convert graph to a line graph so edges become nodes and vice versa
        #edges = nx.closeness_centrality(nx.line_graph(G))

        if street_betw or street_sin:
            edges_clipped = gpd.clip(edges, place)
            return edges_clipped
        return edges

def get_streets_per_bbox(bboxes, network_type='drive', path='nan', intersection_clos=False, street_betw=False, street_sin=False, retain_all=True, save_bbox = "nan"):
    # drive: get drivable public streets (but not service roads)
    # drive_service: get drivable public streets including service roads
    # walk: get all streets and paths that pedestrians can use (this network type ignores one-way
    # directionality by always connecting adjacent nodes with reciprocal directed edges)
    # bike: get all streets and paths that cyclists can use
    # all: download all (non-private) OpenStreetMap streets and paths
    # all_private: download all OpenStreetMap streets and paths, including private-access 
    for bbox in bboxes:
        print(bbox)
        [minx,miny,maxx,maxy] = bbox
        bbox = [maxy, miny, maxx, minx]

        G = ox.graph_from_bbox(*bbox, network_type=network_type, retain_all=retain_all)
        

        # store the street network if no enrichment is needed
        if not intersection_clos and not street_betw and not street_sin:
            edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            return edges

        # if Intersections - Nodes store closeness per intersection
        if intersection_clos:
            nodes = compute_node_closeness(G)
            print("nodes done")
        
        # Streets - Edges
        if street_betw:
            edges = compute_edge_betweenness(G)
            print("edges betweenness done")
        
        if street_betw and street_sin:
            edges['edge_sinuosity'] = edges.apply(lambda row: compute_edge_sinuosity(row), axis=1)

        if street_sin and not street_betw:
            edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            edges['edge_sinuosity'] = edges.apply(lambda row: compute_edge_sinuosity(row), axis=1)
        
        for key in required_keys:
            if key not in edges.keys():
                edges[key] = 'nan'

        if save_bbox != "nan":
            [minx_s,miny_s, maxx_s,maxy_s] = bbox
            save_df = edges.cx[minx_s:maxx_s, miny_s:maxy_s]
        else:
            save_df = edges
        
        if path != "nan" and os.path.exists(path):
            df = pd.read_csv(path)
            # gdf = gpd.GeoDataFrame(df).set_geometry('geometry')
            combined = pd.concat([df, save_df], ignore_index=True)
            print(combined)
            combined.to_csv(path)
        elif path != "nan" and (not os.path.exists(path)):
            save_df.to_csv(path)

        return edges 
    
def get_streets_per_polygon(polygons, network_type='drive', path='nan', intersection_clos=False, street_betw=False, street_sin=False, retain_all=True, save_bbox = "nan"):
    # drive: get drivable public streets (but not service roads)
    # drive_service: get drivable public streets including service roads
    # walk: get all streets and paths that pedestrians can use (this network type ignores one-way
    # directionality by always connecting adjacent nodes with reciprocal directed edges)
    # bike: get all streets and paths that cyclists can use
    # all: download all (non-private) OpenStreetMap streets and paths
    # all_private: download all OpenStreetMap streets and paths, including private-access 
    for polygon in polygons:
        G = ox.graph_from_polygon(polygon, network_type=network_type, retain_all=retain_all)
        

        # store the street network if no enrichment is needed
        if not intersection_clos and not street_betw and not street_sin:
            edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            return edges

        # if Intersections - Nodes store closeness per intersection
        if intersection_clos:
            nodes = compute_node_closeness(G)
            print("nodes done")
        
        # Streets - Edges
        if street_betw:
            edges = compute_edge_betweenness(G)
            print("edges betweenness done")
        
        if street_betw and street_sin:
            edges['edge_sinuosity'] = edges.apply(lambda row: compute_edge_sinuosity(row), axis=1)

        if street_sin and not street_betw:
            edges = ox.utils_graph.graph_to_gdfs(G, nodes=False, edges=True, node_geometry=False, fill_edge_geometry=True)
            edges['edge_sinuosity'] = edges.apply(lambda row: compute_edge_sinuosity(row), axis=1)
        
        for key in required_keys:
            if key not in edges.keys():
                edges[key] = 'nan'

        if save_bbox != "nan":
            save_df = edges
        else:
            save_df = edges
        
        if path != "nan" and os.path.exists(path):
            df = pd.read_csv(path, index_col=0)
            # gdf = gpd.GeoDataFrame(df).set_geometry('geometry')
            combined = pd.concat([df, save_df], ignore_index=True)
            # print(combined)
            combined.to_csv(path)
        elif path != "nan" and (not os.path.exists(path)):
            save_df.to_csv(path)

        return edges 
    
def complete_roads(df_roads):
    # make sure all keys are present to avoid key errors 
    for key in required_keys:
            if key not in df_roads.keys():
                df_roads[key] = 'nan'
    return df_roads

        
        

