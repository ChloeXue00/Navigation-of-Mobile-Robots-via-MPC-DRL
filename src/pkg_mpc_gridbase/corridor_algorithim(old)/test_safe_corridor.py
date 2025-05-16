import os
import math
from typing import Union, Optional

from scipy.spatial import ConvexHull
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import CirclePolygon
from shapely.geometry import Polygon, Point

from basic_map.map_geometric import GeometricMap
from basic_obstacle.geometry_plain import PlainPolygon
from map_processor import MapProcessor
from corridor_algorithim import QuadTree , QuadTreeNode


"""
    QUESTIONS:
        reference_path = [(1, 1), (2, 1), (3, 1), (4, 1), (7, 1),(8, 3), (7, 5), (7, 7)] #reference path in geometric map ?

        distance_threshold = ((map_height + map_width) / 2) * 0.2 # I am not so sure about how large this variable should be 

        check _convert_reference_path_occupancy and reference_path_to_cells, if the idea is correct and where should they be
        can I work with an upside down map?


        what is the min_size ?
        quadtree = QuadTree(inflated_map, min_size=robot_width*200, rescale_geo_occupancy=rescale_geo_occupancy)




        Questions, 
        1. how many values will planner.get_local_ref( return, shouldnt it be 1?
        2. what is target node? what is current target node?
"""


def inflate_map(original_map: GeometricMap, inflation_margin: float):
    boundary_coords, obstacle_coords_list = original_map()
    for i, obs in enumerate(obstacle_coords_list):
        inflated_obs = PlainPolygon.from_list_of_tuples(obs).inflate(inflation_margin)
        obstacle_coords_list[i] = inflated_obs()
    boundary_polygon = PlainPolygon.from_list_of_tuples(boundary_coords).inflate(-inflation_margin)
    boundary_coords = boundary_polygon()
    return GeometricMap.from_raw(boundary_coords, obstacle_coords_list)

def main1():
    # Test data
    robot_width = 0.5
    boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    obstacle_list = [{"id_": 1, "vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]},
                     {"id_": 2, "vertices": [(5, 3), (6, 3), (6, 6), (5, 6), (7,8),(8,8)]},
                     {"id_": 3, "vertices": [(5, 1), (6, 1), (6, 6), (5, 6)]}
                     ]
    """obstacle_list = [{"id_": 1, "vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]},
                   {"id_": 2, "vertices": [(5, 3), (6, 3), (6, 6), (5, 6)]},
                  ]"""
    obstacle_info_list = [GeometricMap.dict_to_obstacle_info(obs) for obs in obstacle_list]
    geometric_map = GeometricMap(boundary, obstacle_info_list)

    #inflation margin of the map
    inflation_margin = robot_width


    # Instantiate MapProcessor
    processor = MapProcessor()

    # Load and inflate map
    processor.load_map(geometric_map, inflation_margin)
    inflated_map = processor._inflated_map
    height, width, _ = inflated_map.shape
    # Generate quadtree
    inflated_map_grayscale = np.dot(inflated_map[..., :3], [0.2989, 0.5870, 0.1140]) / 255.0
    inflated_map_binary = np.where(inflated_map_grayscale > 0.5, 1.0, 0.0)
    rescale_geo_occupancy = 100  # You need to define the appropriate value for your case
    quadtree = QuadTree(inflated_map, min_size=robot_width*200, rescale_geo_occupancy=rescale_geo_occupancy)

    # Convert reference path to occupancy map
    reference_path = [(1, 1), (2, 1), (3, 1), (4, 1), (7, 1),(8, 3), (7, 5),(8,6),(8,8),(2,8)] #reference path in geometric map ?
    #reference_path = [(3, 1), (4, 1), (7, 1)] #reference path in geometric map ?
    #reference_path = [(7, 5),(4,7)] #reference path in geometric map ?
    #reference_path = [(8, 1),(8,7)] #reference path in geometric map ?

    # Create a figure and axis
    fig, ax1 = plt.subplots()

    # Visualize inflated map
    ax1.imshow(inflated_map, cmap='gray')

    # Visualize convex hulls on the path
    #distance_threshold = ((map_height + map_width) / 2)*0.2 # I am not so sure about how large this variable should be 
    distance_threshold = 50
    #reference_cells = quadtree.reference_path_to_cells(geo_map_size=geo_map_size,reference_path=reference_path, distance_threshold=distance_threshold)
    
    convexhulls_intersection_reference = quadtree.collect_convex_hulls_on_path(reference_path, distance_threshold)
    for hull in convexhulls_intersection_reference:
        for simplex in hull.simplices:
            ax1.plot(hull.points[simplex, 0], hull.points[simplex, 1], color='orange')

    # Plot the reference path on the occupancy map
    reference_path_occupancy = quadtree._convert_reference_path_occupancy(reference_path) #in order to plot the line in the same plot as the occupancy map
    ref_path_xs, ref_path_ys = zip(*reference_path_occupancy)
    ax1.plot(ref_path_xs, ref_path_ys, color='red', marker='o')


    #plot geometric corresponedent
        # Create a figure and axis
    fig, ax2 = plt.subplots()
    #geometric_map = inflate_map(geometric_map, inflation_margin)
    # Plot circles for circles designed for the two following points in the reference_path
    #geometric_map.plot(ax2)
    x_min, x_max, y_min, y_max = geometric_map.get_boundary_scope()
    print ("height/heiht geo", height/(y_max-y_min))
    ax2.imshow(inflated_map,extent = [x_min, x_max, y_min, y_max])

    convexhulls_intersection_reference = quadtree.convex_hulls_to_orignal_geoMap(convexhulls_intersection_reference)
    for hull in convexhulls_intersection_reference:
        for simplex in hull.simplices:
            ax2.plot(hull.points[simplex, 0], hull.points[simplex, 1], color='orange')

    ref_path_xs, ref_path_ys = zip(*reference_path)
    ax2.plot(ref_path_xs, ref_path_ys, color='red', marker='o')

    plt.title('Inflated Map with Convex Hulls and Reference Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.subplots_adjust(hspace=0.5)  # Adjust the value as needed for more or less space
    plt.show()




def main2():
    # Test data
    robot_width = 0.5
    boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    obstacle_list = [{"id_": 1, "vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]},
                     {"id_": 2, "vertices": [(5, 3), (6, 3), (6, 6), (5, 6)]}
                     ]
    """obstacle_list = [{"id_": 1, "vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]},
                    {"id_": 2, "vertices": [(5, 3), (6, 3), (6, 6), (5, 6), (8,8)]},
                    {"id_": 3, "vertices": [(5, 1), (6, 1), (6, 6), (5, 6)]}
                    ]"""
    obstacle_info_list = [GeometricMap.dict_to_obstacle_info(obs) for obs in obstacle_list]
    geometric_map = GeometricMap(boundary, obstacle_info_list)

    inflation_margin = robot_width

    # Instantiate MapProcessor
    processor = MapProcessor()

    # Load and inflate map
    processor.load_map(geometric_map, inflation_margin)
    inflated_map = processor._inflated_map

    # Generate quadtree
    rescale_geo_occupancy = 100  # You need to define the appropriate value for your case
    quadtree = QuadTree(inflated_map, min_size=robot_width*100, rescale_geo_occupancy=rescale_geo_occupancy)


    # Visualize inflated map
    #plt.imshow(inflated_map, cmap='gray', origin='lower')
        # Create a figure and axis
    fig, ax = plt.subplots()
    height, width, _ = inflated_map.shape

    # Visualize inflated map
    ax.imshow(inflated_map, cmap='binary')  # Adjust the colormap as needed

    # Visualize convex hulls around each leaf node in the quadtree
    for leaf_node in quadtree.leaf_nodes:
        x, y, subgrid_size, value, level = leaf_node

        # Calculate the coordinates of the subgrid region
        x_end = x + subgrid_size
        y_end = y + subgrid_size

        # Create points representing the subgrid region
        points = np.array([[y, x], [y, x_end], [y_end, x_end], [y_end, x]])

        # Calculate the convex hull of the points
        hull = ConvexHull(points)

        # Assign colors based on the level of the node
        if level == 1:
            color = 'blue'
        elif level == 2:
            color = 'green'
        elif level == 3:
            color = 'magenta'
        elif level == 4:
            color = 'red'
        elif level == 5:
            color = 'black'
        elif level == 6:
            color = 'yellow'
        elif level == 7:
            color = 'cyan'
        else:
            color = 'cyan'
        # Plot the convex hull with the assigned color
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], color=color)

    plt.title('Inflated Map with Convex Hulls')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    

"""
#this Test is if you want to test to see the leaves at a specific level
    #Why would you want it?
    #- sometimes in the orginal test, different levels might overlap entirely, 
    #- you will believe that the leaf at respective level might be missing but infact they are just overlapping entirely
    #- this test proves it
def main():
    # Test data
    robot_width = 0.5
    boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    obstacle_list = [{"id_": 1, "vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]},
                     {"id_": 2, "vertices": [(5, 3), (6, 3), (6, 6), (5, 6)]}
                     ]
    obstacle_info_list = [GeometricMap.dict_to_obstacle_info(obs) for obs in obstacle_list]
    geometric_map = GeometricMap(boundary, obstacle_info_list)

    inflation_margin = robot_width

    # Instantiate MapProcessor
    processor = MapProcessor()

    # Load and inflate map
    processor.load_map(geometric_map, inflation_margin)
    inflated_map = processor._inflated_map

    # Generate quadtree
    rescale_geo_occupancy = 100  # You need to define the appropriate value for your case
    quadtree = QuadTree(inflated_map, min_size=robot_width*200, rescale_geo_occupancy=rescale_geo_occupancy)

    # Visualize inflated map
    plt.imshow(inflated_map, cmap='gray', origin='lower')

    # Visualize convex hulls around leaf nodes at level 3 in the quadtree
    for leaf_node in quadtree.leaf_nodes:
        x, y, subgrid_size, value, level = leaf_node

        # Check if the leaf node is at level 3
        if level == 2:
            # Calculate the coordinates of the subgrid region
            x_end = x + subgrid_size
            y_end = y + subgrid_size

            # Create points representing the subgrid region
            points = np.array([[y, x], [y, x_end], [y_end, x_end], [y_end, x]])

            # Calculate the convex hull of the points
            hull = ConvexHull(points)

            # Assign colors based on the level of the node
            color = 'magenta'

            # Plot the convex hull with the assigned color
            for simplex in hull.simplices:
                plt.plot(points[simplex, 0], points[simplex, 1], color=color)

    plt.title('Inflated Map with Convex Hulls at Level 3')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()"""

def main3():
    #from connective_quadtree import QuadTree , QuadTreeNode
    """testing the update_Quadtree
    """
    robot_width = 0.5
    boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    obstacle_list = [{"id_": 1, "vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]},
                     ]

    obstacle_info_list = [GeometricMap.dict_to_obstacle_info(obs) for obs in obstacle_list]
    geometric_map = GeometricMap(boundary, obstacle_info_list)

    inflation_margin = robot_width

    # Instantiate MapProcessor
    processor = MapProcessor()

    # Load and inflate map
    processor.load_map(geometric_map, inflation_margin)
    inflated_map = processor._inflated_map

    # Generate quadtree
    rescale_geo_occupancy = 100  # You need to define the appropriate value for your case
    quadtree = QuadTree(inflated_map, min_size=robot_width*30, rescale_geo_occupancy=rescale_geo_occupancy)


    # Visualize inflated map
    #plt.imshow(inflated_map, cmap='gray', origin='lower')
        # Create a figure and axis
    fig, ax = plt.subplots()
    height, width, _ = inflated_map.shape

    # Visualize inflated map
    ax.imshow(inflated_map, cmap='binary')  # Adjust the colormap as needed

    # Visualize convex hulls around each leaf node in the quadtree
    for leaf_node in quadtree.leaf_nodes:
        x, y, subgrid_size, value, level = leaf_node
        print(y,x, level)
        # Calculate the coordinates of the subgrid region
        x_end = x + subgrid_size
        y_end = y + subgrid_size 

        # Create points representing the subgrid region
        #points = np.array([[x, y], [x_end, y], [x_end, y_end], [x, y_end]])
        points = np.array([[y, x], [y, x_end], [y_end, x_end], [y_end, x]])

        # Calculate the convex hull of the points
        hull = ConvexHull(points)

        # Assign colors based on the level of the node
        if level == 1:
            color = 'blue'
        elif level == 2:
            color = 'green'
        elif level == 3:
            color = 'magenta'
        elif level == 4:
            color = 'red'
        elif level == 5:
            color = 'black'
        elif level == 6:
            color = 'yellow'
        elif level == 7:
            color = 'cyan'
        else:
            color = 'cyan'
        # Plot the convex hull with the assigned color
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], color=color)

    plt.title('Inflated Map with Convex Hulls')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show() 

    #part for map 2

    obstacle_list = [{"id_": 1, "vertices": [(2, 8), (3, 8), (3, 9), (2, 9)]},
                     {"id_": 2, "vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]}
                    ]

    obstacle_info_list = [GeometricMap.dict_to_obstacle_info(obs) for obs in obstacle_list]
    geometric_map = GeometricMap(boundary, obstacle_info_list)

    inflation_margin = robot_width

    # Instantiate MapProcessor
    processor = MapProcessor()

    # Load and inflate map
    processor.load_map(geometric_map, inflation_margin)
    inflated_map2 = processor._inflated_map

    # Visualize inflated map
    #plt.imshow(inflated_map, cmap='gray', origin='lower')
        # Create a figure and axis
    fig, ax2 = plt.subplots()
    height, width, _ = inflated_map2.shape

    # Visualize inflated map
    ax2.imshow(inflated_map2, cmap='binary')  # Adjust the colormap as needed
    quadtree.update_Quadtree(inflated_map2)

    # Visualize convex hulls around each leaf node in the quadtree
    for leaf_node in quadtree.leaf_nodes:
        x, y, subgrid_size, value, level = leaf_node

        # Calculate the coordinates of the subgrid region
        x_end = x + subgrid_size
        y_end = y + subgrid_size

        # Create points representing the subgrid region
        #points = np.array([[x, y], [x_end, y], [x_end, y_end], [x, y_end]])
        points = np.array([[y, x], [y, x_end], [y_end, x_end], [y_end, x]])

        # Calculate the convex hull of the points
        hull = ConvexHull(points)

        # Assign colors based on the level of the node
        if level == 1:
            color = 'blue'
        elif level == 2:
            color = 'green'
        elif level == 3:
            color = 'magenta'
        elif level == 4:
            color = 'red'
        elif level == 5:
            color = 'black'
        elif level == 6:
            color = 'yellow'
        elif level == 7:
            color = 'cyan'
        else:
            color = 'cyan'
        # Plot the convex hull with the assigned color
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], color=color)

    plt.title('Inflated Map2 with Convex Hulls')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def inflate_map(original_map: GeometricMap, inflation_margin: float):
    boundary_coords, obstacle_coords_list = original_map()
    for i, obs in enumerate(obstacle_coords_list):
        inflated_obs = PlainPolygon.from_list_of_tuples(obs).inflate(inflation_margin)
        obstacle_coords_list[i] = inflated_obs()
    boundary_polygon = PlainPolygon.from_list_of_tuples(boundary_coords).inflate(-inflation_margin)
    boundary_coords = boundary_polygon()
    return GeometricMap.from_raw(boundary_coords, obstacle_coords_list)


def main4():
    # Test data
    robot_width = 0.5
    boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    obstacle_list = [{"id_": 1, "vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]},
                     {"id_": 2, "vertices": [(5, 3), (6, 3), (6, 6), (5, 6), (8,8)]},
                     {"id_": 3, "vertices": [(5, 1), (6, 1), (6, 6), (5, 6)]}
                     ]
    #obstacle_list = [{"id_": 1, "vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]},
                   # {"id_": 2, "vertices": [(5, 3), (6, 3), (6, 6), (5, 6)]},
                  # ]
    obstacle_info_list = [GeometricMap.dict_to_obstacle_info(obs) for obs in obstacle_list]
    geometric_map = GeometricMap(boundary, obstacle_info_list)
    # Get the size of the geometric map
    x_min, x_max, y_min, y_max = geometric_map.get_boundary_scope()

    # Calculate the size
    map_width = x_max - x_min
    map_height = y_max - y_min
    geo_map_size = (int(map_width), int(map_height))

    #inflation margin of the map
    inflation_margin = robot_width


    # Instantiate MapProcessor
    processor = MapProcessor()

    # Load and inflate map
    processor.load_map(geometric_map, inflation_margin)
    inflated_map = processor._inflated_map
    height, width, _ = inflated_map.shape

    # Generate quadtree
    inflated_map_grayscale = np.dot(inflated_map[..., :3], [0.2989, 0.5870, 0.1140]) / 255.0
    inflated_map_binary = np.where(inflated_map_grayscale > 0.5, 1.0, 0.0)
    rescale_geo_occupancy = 100  # You need to define the appropriate value for your case
    quadtree = QuadTree(inflated_map_binary, min_size=robot_width*200, rescale_geo_occupancy=rescale_geo_occupancy)

    # Convert reference path to occupancy map
    reference_path = [(1, 1), (2, 1), (3, 1), (4, 1), (7, 1),(8, 3), (7, 5),(8,6),(8,8),(2,8)] #reference path in geometric map ?
    #reference_path = [(3, 1), (4, 1), (7, 1)] #reference path in geometric map ?
    #reference_path = [(7, 5),(4,7)] #reference path in geometric map ?
    #reference_path = [(8, 1),(8,7)] #reference path in geometric map ?

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Visualize inflated map
    ax.imshow(inflated_map, cmap='gray')

    # Visualize convex hulls on the path
    #distance_threshold = ((map_height + map_width) / 2)*0.2 # I am not so sure about how large this variable should be 
    distance_threshold = 110
    #reference_cells = quadtree.reference_path_to_cells(geo_map_size=geo_map_size,reference_path=reference_path, distance_threshold=distance_threshold)
    
    convexhulls_intersection_reference = quadtree.collect_convex_hulls_on_path(reference_path, height, distance_threshold)
    for key, hull in convexhulls_intersection_reference.items():
        for simplex in hull.simplices:
            ax.plot(hull.points[simplex, 0], hull.points[simplex, 1], color='orange')
        hull_center = np.mean(hull.points[hull.vertices], axis=0)
        ax.text(hull_center[0], hull_center[1], key, color='orange')

    # Plot the reference path on the occupancy map
    reference_path_occupancy = quadtree._convert_reference_path_occupancy(reference_path,height) #in order to plot the line in the same plot as the occupancy map
    ref_path_xs, ref_path_ys = zip(*reference_path_occupancy)
    ax.plot(ref_path_xs, ref_path_ys, color='red', marker='o')


    # Plot circles for circles designed for the two following points in the reference_path

    plt.title('Inflated Map with Convex Hulls and Reference Path')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()





if __name__ == "__main__":
    main1()
    #main2()
    #main3()
    #main4()

