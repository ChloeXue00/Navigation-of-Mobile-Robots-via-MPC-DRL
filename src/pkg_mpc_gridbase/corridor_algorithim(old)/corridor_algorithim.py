
"""
The algorithm for this code is:
- Function to create an octree from an occupancy grid
- Divide the areas of the occupancy map
- Check if there are obstacles in the area, if not then it is a leaf node
- Otherwise, divide the area into 4 subareas and repeat the process
- Until all leaf nodes are found (flag them as green) or the maximum depth is reached 
    (maximum size of the subarea) (flag them as red)
- Collect all the leaf nodes which are not occupied and use them as the nodes for 
    the A* search algorithm later

"""

from typing import Optional

import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Point,LineString,Polygon


class QuadTree:

    """
    QuadTree algorithm implementation.
    """

    def __init__(self, occupancy_grid: np.ndarray, min_size:float, rescale_geo_occupancy:int =100):
        """
        Initialize the QuadTree.

        Args:
            occupancy_grid (ndarray): The occupancy grid.
            min_size (int): The minimum size for splitting.
            rescale_geo_occupancy (float): Rescale factor for geographic occupancy.
        """
        self.min_size = min_size
        self.occupancy_grid = occupancy_grid
        self.occupancy_height, self.occupancy_width, _ = occupancy_grid.shape
        self.root = QuadTreeNode(occupancy_grid, min_size, 0, 0, 0)
        self.leaf_nodes : list[tuple] = [] #tuple of QuadTreeNode information
        self.convex_hulls : list[ConvexHull]  = []
        self._collect_leaf_nodes(self.root)
        self._create_convex_hulls()
        self.rescale_geo_occupancy = rescale_geo_occupancy

    def _collect_leaf_nodes(self, node: 'QuadTreeNode'):
        """
        Recursively collect leaf nodes.

        Args:
            node (QuadTreeNode): The current node.
        """
        if node.is_leaf:
            if np.all(node.value == [255, 255, 255]):
                self.leaf_nodes.append((node.x, node.y, node.subgrid_size, node.value, node.level))
        else:
            for child in node.children:
                self._collect_leaf_nodes(child)

    def _create_convex_hulls(self):
        """Create convex hulls for leaf nodes."""
        for leaf_node in self.leaf_nodes:
            x, y, subgrid_size, value, level = leaf_node
            # Calculate the coordinates of the subgrid region
            x_end = x + subgrid_size
            y_end = y + subgrid_size
            # Create points representing the subgrid region
            points = np.array([[y, x], [y, x_end], [y_end, x_end], [y_end, x]])
            # Calculate the convex hull of the points
            hull = ConvexHull(points)
            self.convex_hulls.append(hull)

    def update_Quadtree(self, new_occupancyMap: np.ndarray):
            """
            Update the quadtree based on the new occupancy map.

            Args:
                new_occupancyMap (ndarray): The new occupancy map.


            
        You need to have a function, which takes in the previous tree. and the current new version of the occupancy grid
        - checks if all the white nodes are still white, if not break them even further 
        - check for all the black nodes ares still black, if they are not make them white
        * A possible improvement, is to group these new white nodes (since they are small) into a larger zone

            """
            # Traverse existing leaf nodes and check against new occupancy map
            old_leaf_nodes = self.leaf_nodes
            for leaf_node in (old_leaf_nodes):
                x, y, subgrid_size, value, level = leaf_node
                #print(x,y)
                # Extract corresponding part from the new occupancy map
                new_subgrid = new_occupancyMap[x : x + subgrid_size, y : y + subgrid_size]
                
                # Check if any value is 0 (indicating a new obstacle)
                if np.any(new_subgrid == [0, 0, 0]):
                    self.leaf_nodes.remove(leaf_node)
                    new_node = QuadTreeNode(new_subgrid, self.min_size,x, y, level )
                    self._collect_leaf_nodes(new_node)
                    
                    

    



    def _convert_reference_path_occupancy(self, reference_path):
        """
        Convert reference path coordinates to occupancy map coordinates.

        Args:
            reference_path (ndarray): Array representing the reference path.
            height_occupancy (int): Height of the occupancy map.

        Returns:
            list: List of converted reference path coordinates.
        """
        path_in_occupancy_map = []
        for cell in reference_path:
            scaled_cell = (
                float(cell[0] * self.rescale_geo_occupancy),
                float(self.occupancy_height - (cell[1] * self.rescale_geo_occupancy)),
            )
            path_in_occupancy_map.append(scaled_cell)
        return path_in_occupancy_map


    def collect_convex_hulls_on_path(self, reference_path, threshold):
        # Convert reference cells to occupancy map coordinates
        reference_cells_occupancy = self._convert_reference_path_occupancy(reference_path)
        
        # Create a LineString from the reference cells
        reference_line = LineString(reference_cells_occupancy)
        
        # List to store convex hulls intersecting with the reference line
        path_convex_hulls = []
        
        for hull in self.convex_hulls:
            # Check if the reference line intersects with the convex hull
            if Polygon(hull.points[hull.vertices]).intersects(reference_line):
                path_convex_hulls.append(hull)
            
            else:
                # Check distance between each vertex of the convex hull and the reference line
                for vertex in hull.points[hull.vertices]:
                    distance = reference_line.distance(Point(vertex[0], vertex[1]))
                    if distance < threshold:
                        path_convex_hulls.append(hull)
                        break
        return path_convex_hulls

    def convex_hulls_to_orignal_geoMap(self, Convex_Occupancy: list[ConvexHull]):
        """
        Convert convex hulls expressed in occupancy map to geometric map scale.

        Args:
            Convex_Occupancy (list[ConvexHull]): List of convex hulls expressed in occupancy map.
            height_occupancy (int): Height of the occupancy map.

        Returns:
            list[ConvexHull]: List of convex hulls expressed in geometric map scale.
        """
        Convex_Geometric = []  # Initialize a new list to store convex hulls in geometric map scale
        print(Convex_Occupancy)
        for hull in Convex_Occupancy:
            # Initialize a new list to store the modified vertex coordinates
            vertices_geo = []
            # Convert each vertex coordinates from occupancy map scale to geometric map scale
            for vertex in hull.points:
                # Convert vertex coordinates from occupancy map scale to geometric map scale
                x_geo = vertex[1] / self.rescale_geo_occupancy
                y_geo = self.occupancy_height/self.rescale_geo_occupancy - (vertex[0] / self.rescale_geo_occupancy)
                # Append the modified vertex coordinates to the new list
                vertices_geo.append([x_geo, y_geo])
            # Create a new ConvexHull object with the modified vertices
            convex_geo = ConvexHull(vertices_geo)
            # Append the convex hull expressed in geometric map scale to the new list
            Convex_Geometric.append(convex_geo)
        return Convex_Geometric

class QuadTreeNode:
    """
    QuadTreeNode implementation.
    """

    def __init__(self, occupancy_grid:np.ndarray, min_size : float, x:int, y:int, level:int):
        """
        Initialize the QuadTreeNode.

        Args:
            occupancy_grid (ndarray): The occupancy grid.
            min_size (float): The minimum size for splitting.
            x (int): The x coordinate.
            y (int): The y coordinate.
            level (int): The level of the node.
        """
        self.is_leaf = True
        self.value = occupancy_grid
        self.children : list[QuadTreeNode] = [] 
        self.subgrid_size = occupancy_grid.shape[0]
        self.x = x
        self.y = y
        self.min_size = min_size
        self.level = level

        if self.subgrid_size > min_size and np.any(self.value == [0, 0, 0]):
            self.split()

    def split(self):
        """
        Split the node into sub-nodes.
        """
        size = self.subgrid_size // 2
        for i in range(2):
            for j in range(2):
                child_grid = self.value[i * size : (i + 1) * size, j * size : (j + 1) * size]
                child_node = QuadTreeNode(
                    child_grid, self.min_size, self.x + i * size, self.y + j * size, self.level + 1
                )
                self.children.append(child_node)
        self.is_leaf = False

 














