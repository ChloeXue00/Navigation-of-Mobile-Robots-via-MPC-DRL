
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


#TODO
#UPDATE FUNCTION NEEDS TO BE IMPLEMENTED AS DISCUSSED
# CONNECTIVITY GRAPH NEEDS TO BE IMPLEMENTED 

import math
from enum import Enum
from typing import Optional,Union, Tuple

import numpy as np
from scipy.spatial import ConvexHull # type: ignore
from shapely.geometry import Point, Polygon, LineString # type: ignore
import casadi.casadi as cs # type: ignore


PathNode = tuple[float, float]


class TreeNodeState(Enum):
    """Enum for the state of a tree node."""
    UNKNOWN = 0
    MIX = 1    #has both white and black
    FULL = 2   #reached maximium length
    FREE = 3   

class TreeNodeType(Enum):
    """Enum for the type of a tree node."""
    UNKNOWN = 0
    START = 1
    GOAL = 2
    PATH = 3
    DEADEND = 4



class ScaleOffsetReverseTransform:
    def __init__(self, scale:float=1, offsetx_after:float=0, offsety_after:float=0, x_reverse=False, y_reverse=False, x_max_before=0, y_max_before=0):
        """Transform the given coordinates by some scales and offsets.
        Args:
            scale       : Scale factor.
            offset_after: The offset of x and y axes.
            reverse     : If the x and y axes should be reversed.
            max_before  : The maximal values along x and y axes, used to calculate the reversed coordinates.
        Comments:
            For orginal coordinates z=[x,y], if x or y is reversed, calculate the reversed coordinate first.
            Then calculate the transformed coordinates according to the scaling and the offset.
        """
        self.k = [scale, scale]
        self.b = [offsetx_after, offsety_after]
        self.xr = x_reverse
        self.yr = y_reverse
        self.xm = x_max_before
        self.ym = y_max_before

    def __call__(self, state:Union[list, np.ndarray], forward=True) -> Union[list, np.ndarray]:
        '''Return the transformed state. If forward=False, it means from the transformed state to the original one.
        '''
        tr_state = state.copy()
        if forward:
            if self.xr:
                tr_state[0] = self.xm - tr_state[0]
            if self.yr:
                tr_state[1] = self.ym - tr_state[1]
            tr_state[0] = tr_state[0]*self.k[0]+self.b[0]
            tr_state[1] = tr_state[1]*self.k[1]+self.b[1]
        else:
            tr_state[0] = (state[0]-self.b[0]) / self.k[0]
            tr_state[1] = (state[1]-self.b[1]) / self.k[1]
            if self.xr:
                tr_state[0] = self.xm - tr_state[0]
            if self.yr:
                tr_state[1] = self.ym - tr_state[1]
        return tr_state

    def cvt_coord_x(self, x:np.ndarray, forward=True) -> np.ndarray:
        if forward:
            if self.xr:
                x = self.xm - x
            cvt_x = self.k[0]*x + self.b[0]
        else:
            cvt_x = (x-self.b[0]) / self.k[0]
            if self.xr:
                cvt_x = self.xm - cvt_x
        return cvt_x

    def cvt_coord_y(self, y:np.ndarray, forward=True) -> np.ndarray:
        if forward:
            if self.yr:
                y = self.ym - y
            cvt_y = self.k[1]*y + self.b[1]
        else:
            cvt_y = (y-self.b[1]) / self.k[1]
            if self.yr:
                cvt_y = self.ym - cvt_y
        return cvt_y

    def cvt_coords(self, x:np.ndarray, y:np.ndarray, forward=True) -> np.ndarray:
        '''Return transformed/original coordinates, in shape (2*n).
        '''
        cvt_x = self.cvt_coord_x(x, forward)
        cvt_y = self.cvt_coord_y(y, forward)
        return np.hstack((cvt_x[:,np.newaxis], cvt_y[:,np.newaxis]))




class QuadTree:
    def __init__(self, value_matrix: np.ndarray, min_cell_size: int, num_splits:int=2, rescale_geo_occupancy:int =100):
        """Initialize the QuadTree.

        Args:
            value_matrix: The matrix containing the values of the occupancy grid. Must be two-dimensional.
            min_cell_size: The minimum size for splitting.
            num_splits: The number of splits in each dimension. Default is 2.
        """
        if len(value_matrix.shape) != 2:
            raise ValueError(f"[{self.__class__.__name__}] value_matrix must be two-dimensional, got {len(value_matrix.shape)}")

        self._max_depth = self._get_max_depth(value_matrix, min_cell_size, num_splits)

        self.root = QuadTreeNode.init_root(value_matrix, self.max_depth, auto_split=True)
        
        self.leaf_nodes : list[QuadTreeNode] = []
        self._collect_free_leaf_nodes(self.root)
        
        # TODO: Implement the following methods
        # self.convex_hulls : dict[tuple, ConvexHull]  = {}
        # self._create_convex_hulls()

    @property
    def max_depth(self) -> int:
        """Get the maximum depth of the quadtree."""
        return self._max_depth

    def _get_max_depth(self, value_matrix: np.ndarray, min_cell_size: int, num_splits:int=2):
        """Get the maximum depth of the quadtree.

        Args:
            value_matrix: The matrix containing the values of the occupancy grid.
            min_cell_size: The minimum size for splitting.
            num_splits: The number of splits in each dimension. Default is 2.

        Returns:
            max_depth: The maximum depth of the quadtree.
        """
        max_dim = max(value_matrix.shape)
        max_depth = math.floor(math.log(max_dim/min_cell_size)/math.log(num_splits)) 
        return max_depth
    
    def _collect_free_leaf_nodes(self, node: "QuadTreeNode"):
        """Recursively collect leaf nodes.

        Args:
            node: The current node.
        """
        if node.is_leaf:
            if node.state == TreeNodeState.FREE:
                self.leaf_nodes.append(node)
        else:
            for child in node.children:
                self._collect_free_leaf_nodes(child)

    # def _create_convex_hulls(self):
    #     """Create convex hulls for leaf nodes.""" # TODO this is logically wrong
    #     # raise NotImplementedError
    #     for leaf_node in self.leaf_nodes:
    #         x, y, subgrid_size, values, id_sequence = leaf_node
    #         # Calculate the coordinates of the subgrid region
    #         x_end = x + subgrid_size
    #         y_end = y + subgrid_size
    #         # Create points representing the subgrid region
    #         points = np.array([[y, x], [y, x_end], [y_end, x_end], [y_end, x]])  #为什么这里x 和y 是反的
    #         # points = np.array([[x, y], [x_end, y], [x_end, y_end], [x, y_end]])
    #         # Calculate the convex hull of the points
    #         hull = ConvexHull(points)
    #         self.convex_hulls[id_sequence] = hull
    def _create_convex_hulls(self):
        """Create convex hulls for leaf nodes.
        
        This method creates convex hulls for all free leaf nodes in the quadtree.
        The convex hulls are stored in the self.convex_hulls dictionary with the
        node's id_sequence as the key.
        """
        self.convex_hulls = {}  # 确保字典被初始化
        
        for leaf_node in self.leaf_nodes:
            # 使用节点的正确属性获取顶点
            vertices = np.array(leaf_node.vertices)
            
            # 检查顶点是否足够构成凸包（至少3个点）
            if len(vertices) >= 3:
                # 计算凸包
                hull = ConvexHull(vertices)
                # 存储凸包，使用节点的id_sequence作为键
                self.convex_hulls[leaf_node.id_sequence] = hull
            else:
                # 如果点不足以形成凸包，可以跳过或记录警告
                print(f"Warning: Node {leaf_node.id_sequence} has insufficient vertices for convex hull")

    # def update_Quadtree(self, new_occupancyMap: np.ndarray):
        
    #     #how do you put this dynamic obstacle in the geometric map, or should we tackle the problem on a higher level like collecting the nodes that intersect a refernce path 
    #     #should we use the position of the dynamic obstacle to only get the treenodes close to it, or should we just get a new camera feed and update everything
        # 1. 更新 Occupancy Map
        # 动态障碍的更新可以通过以下两种方式：

        # 直接替换：完全用 new_occupancyMap 替换当前地图。适合全局更新（例如，新的传感器输入）。
        # 局部更新：仅更新与动态障碍相关的区域，减少计算量。

    #     #if we use the position, how do we transfer position to the correct node
    #     #we also have two problems, when one node get split its older neighbors dont get notified. However, even if they were to be updated current setup does not work for many neighbors on the same side 
    #     #nor the picture case
        # 2. 区域化更新
        #     针对动态障碍，只更新靠近障碍的节点：

        #     定位障碍相关节点：
        #     使用障碍位置（或传感器提供的区域）定位受影响的 Quadtree 节点。
        #     可以通过递归遍历或几何计算定位。
        #     只更新相关节点：
        #     对障碍所在区域的节点进行分裂或合并。
        #     更新时保留其他部分不变。

    #     #if the obstacle moves how do we merge back the neighbours while maintaining the square shape of the nodes
        # 3. 邻居关系更新
        #     当节点分裂或合并时，需要重新计算其邻居：

        #     分裂时： 新生成的子节点需要更新其邻居关系。
        #     合并时： 合并后的节点需更新为其邻居的单一节点。
        
    #     #say everything works, what if the dynamic obstacle comes exactly in the middle of the reference path, and it is moving faster than what
    #     #the robot is trying to avoid the obstacle 
        # 4. 动态障碍的特殊情况
        #     处理动态障碍穿过路径的情况：

        #     如果动态障碍正好位于参考路径的中间：
        #     动态规划器（如 MPC 或 DRL）需要即时重新规划路径。
        #     可考虑启发式算法绕过障碍。
        #     障碍速度快于机器人：
        #     提前预测障碍物的移动（结合 Kalman Filter 等）。
        #     适当缩短规划的时间窗口，提高响应速度。

    #     #say everything works how do we run this with accounting to the time horizon of 20, do we run update the map then go for 20 time horizon from the solver
    #     #then move update the map call the solver for 20 more horizon steps. 

        

    #     # TODO Update this code according to the new structure
    #     raise NotImplementedError

    # """Update the occupancy grid with new data. 全局更新"""

    # def update_occupancy_map(self, new_occupancy_map: np.ndarray):
    #     if new_occupancy_map.shape != self.root.value_matrix.shape:
    #         raise ValueError("New occupancy map must match the current map dimensions.")

    #     # Replace the root node's value matrix with the new map
    #     self.root.value_matrix = new_occupancy_map

    #     # Re-split the root if necessary
    #     self.root.auto_split(max_depth=self._max_depth)
    #     self._collect_free_leaf_nodes(self.root)

    """
    Update the Quadtree based on a dynamic obstacle's position and size.

    Args:
        obstacle_position: (x, y) position of the dynamic obstacle.
        obstacle_radius: Radius of the obstacle's influence.
    """
    def update_quadtree_with_obstacle(self, obstacle_position: tuple, obstacle_radius: float):
  
        def is_node_affected(node, obstacle_position, obstacle_radius):
            # Check if the node intersects with the obstacle's influence radius
            x_min, x_max, y_min, y_max = node.get_bounds()
            x, y = obstacle_position
            return (
                x_min - obstacle_radius <= x <= x_max + obstacle_radius and
                y_min - obstacle_radius <= y <= y_max + obstacle_radius
            )

        def recursive_update(node):
            if is_node_affected(node, obstacle_position, obstacle_radius):
                if not node.is_leaf():
                    # Recursively check children
                    for child in node.children:
                        recursive_update(child)
                else:
                    # If it's a leaf node and affected, decide whether to split or mark as occupied
                    if node.size > self.min_cell_size:
                        node.split()
                    else:
                        node.mark_occupied()  # Mark the node as occupied
            elif node.is_leaf():
                # Merge unaffected nodes if previously split
                node.merge()

        recursive_update(self.root)
        self._collect_free_leaf_nodes(self.root)
     
    # 邻居关系更新   
    def update_neighbors(node):
        """
        Update the neighbor relationships for a given node.
        """
        for neighbor in node.neighbors:
            if neighbor.is_leaf():
                # Update neighbor relationship based on current structure
                neighbor.neighbors = [n for n in neighbor.get_adjacent_nodes()]
    def dynamic_update_and_plan(self, obstacle_positions, time_horizon):
        """
        Integrate map update with planning steps.
        
        Args:
            obstacle_positions: List of dynamic obstacle positions.
            time_horizon: Number of steps to plan ahead.
        """
        for t in range(time_horizon):
            # Update map with dynamic obstacles
            for obstacle_pos in obstacle_positions:
                self.update_quadtree_with_obstacle(obstacle_pos, obstacle_radius=1.0)
            
            # Update safe areas
            self._create_convex_hulls()
            
            # Call the planner (e.g., MPC/DRL) to generate the next steps
            next_actions = self.planner.plan(self.leaf_nodes, convex_hulls=self.convex_hulls)
            
            # Execute one step and repeat
            self.robot.execute(next_actions[0])

    #根据一条参考路径（reference path），提取靠近这条路径的空闲叶子节点（leaf nodes），用于后续路径规划、障碍避让或可行区域生成。
    def collect_tree_nodes_on_path(self, reference_path: list[PathNode], height_occupancy: int, threshold: float, rescale_geo_occupancy: int, rescale_geo_inflated_occupancy:float):
        """
        Collect tree nodes along a reference path.

        Args:
            reference_path (list): List of PathNodes representing the reference path.
            height_occupancy (int): Height of the occupancy map.
            threshold (float): Threshold distance for considering a tree node.
            rescale_geo_occupancy (int): Rescale factor for the geometric map scale.
            rescale_geo_inflated_occupancy (float): Rescale factor for the inflated geometric map scale.

        Returns:
            path_tree_nodes (dict): Dictionary containing tree nodes found along the reference path.
            geo_map (dict): Dictionary containing convex hulls expressed in the original geometric map scale.

        Comments:
            This function collects tree nodes along a reference path based on a given threshold distance.
            It first converts the reference path to occupancy map coordinates using the provided rescale factor.
            Then, it iterates through the leaf nodes of the quadtree and calculates the distance between each leaf node
            and the reference path. If the distance is less than the threshold, the leaf node is considered as part of
            the path. The function returns a dictionary containing the tree nodes found along the reference path,
            along with convex hulls expressed in the original geometric map scale.
        """
        reference_cells_occupancy = self.convert_reference_path_occupancy(reference_path, height_occupancy=height_occupancy, rescale_geo_occupancy=rescale_geo_occupancy)
        path_tree_nodes: dict[tuple, QuadTreeNode] = {}
        for ln in self.leaf_nodes:
            poly = Polygon(ln.vertices)
            distance = LineString(reference_cells_occupancy).distance(poly)
            if distance < threshold:
                path_tree_nodes[ln.id_sequence] = ln
        return path_tree_nodes, self.path_tree_nodes_to_orignal_geoMap(path_tree_nodes, rescale_geo_inflated_occupancy, height_occupancy)
    
    #全局坐标系的转换
    def path_tree_nodes_to_orignal_geoMap(self, path_tree_nodes: dict[tuple, "QuadTreeNode"], rescale_geo_inflated_occupancy: float, height_occupancy:int ):
        """
        Convert convex hulls expressed in path tree nodes to the original geometric map scale.

        Args:
            path_tree_nodes (dict): Dictionary containing path tree nodes with their keys and corresponding QuadTreeNodes.
            rescale_geo_inflated_occupancy (float): Rescale factor for the inflated geometric map scale.
            height_occupancy (int): Height of the occupancy map.

        Returns:
            Convex_Geometric (dict): Dictionary containing convex hulls expressed in the original geometric map scale.

        Comments:
            This function converts convex hulls expressed in path tree nodes to the original geometric map scale.
            It iterates through the given path tree nodes, retrieves their vertices, and applies a reverse transformation
            to convert them back to the original scale. The transformation includes scaling, offset adjustments, and,
            if specified, reversing the y-axis. Finally, it computes convex hulls for the transformed vertices and stores
            them in a dictionary with the corresponding keys.
        """
        Convex_Geometric :dict[tuple,Tuple[ConvexHull, np.ndarray]] = {} # Initialize a new dictionary to store convex hulls in geometric map scale
        for key, tn in path_tree_nodes.items():
            vertices = np.array(tn.vertices)
            x_vertices = vertices[:, 0]
            y_vertices = vertices[:, 1]
            # Create a transformation object to reverse the scaling and offset
            transform = ScaleOffsetReverseTransform(scale=rescale_geo_inflated_occupancy, offsetx_after=0, offsety_after=0, x_reverse=False, y_reverse=True, x_max_before=height_occupancy/rescale_geo_inflated_occupancy, y_max_before=height_occupancy/rescale_geo_inflated_occupancy)
            # Apply reverse transformation to convert vertices back to original scale
            transformed_vertices = transform.cvt_coords(x_vertices, y_vertices, forward=False)
            # Compute convex hull for transformed vertices
            hull = ConvexHull(transformed_vertices)
            # Store convex hull along with transformed vertices in the dictionary
            Convex_Geometric[key] = (hull , transformed_vertices)
        return Convex_Geometric


    #节点级别凸包运算，返回的凸包列表可以直接用于路径规划、安全区域描述或导航约束。
    def path_tree_nodes_convexhull(self, path_tree_nodes: dict[tuple, "QuadTreeNode"]):
        path_convex_hulls = []

        for key, tn in path_tree_nodes.items():
            vertices = np.array(tn.vertices)
            # Compute convex hull for transformed vertices
            hull = ConvexHull(tn.vertices)
            # Store convex hull along with transformed vertices in the dictionary
            path_convex_hulls.append(hull)
        return path_convex_hulls


    @staticmethod
    def convert_reference_path_occupancy(reference_path, height_occupancy, rescale_geo_occupancy: int):
        """Convert reference path coordinates to occupancy map coordinates.

        Args:
            reference_path (ndarray): Array representing the reference path.
            height_occupancy (int): Height of the occupancy map.

        Returns:
            path_in_occupancy_map: List of converted reference path coordinates.
        """
        path_in_occupancy_map = []
        for cell in reference_path:
            scaled_cell = (
                int(cell[0] * rescale_geo_occupancy),
                int(height_occupancy - (cell[1] * rescale_geo_occupancy)),
            )
            path_in_occupancy_map.append(scaled_cell)
        return path_in_occupancy_map
 

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
            old_leaf_nodes = self.leaf_nodes
            indx=0
            while indx < len(old_leaf_nodes):
                all_free = 0
                leaf_node = old_leaf_nodes[indx]

                i, j = leaf_node._ru_indices
                # Extract corresponding part from the new occupancy map
                new_subgrid = new_occupancyMap[i : i + leaf_node.height, j : j + leaf_node.width]
                
                # Check if any value is 0 (indicating a new obstacle)
                if np.any(new_subgrid == [0,0,0]) :
                    if (len(leaf_node.id_sequence) <  self._max_depth): #occupied
                        index_to_replace = indx
                        old_leaf_nodes.remove(leaf_node)
                        #print("parent to be removed",j,i)
                        # # Remove the value to be replaced
                        leaf_node.split(self._max_depth)
                        old_leaf_nodes[index_to_replace: index_to_replace] = leaf_node.children
                        # print("number of leaf nodes", len(old_leaf_nodes) )
                        # for child in leaf_node.children:
                        #     i, j = child._ru_indices
                        #     print ("child", j,i)
                            #old_leaf_nodes.append(child)
                    else:
                        #print("index to be removed",j,i)
                        old_leaf_nodes.remove(leaf_node)  # Remove the node from leaf_nodes if it cant be split, which means you count it as taken
                        #print("we are removing", len(old_leaf_nodes) )
                else:
                    indx +=1
                # else:
                #     all_free +=1

                # if all_free == 4:
                #     old_leaf_nodes
                #     del old_leaf_nodes[indx-3:indx+1]
            
            self.leaf_nodes=old_leaf_nodes
  



class QuadTreeNode:
    def __init__(self, value_matrix: np.ndarray, ru_i: int, ru_j: int, max_depth: int,
                 id_sequence: tuple[int, ...], neighbors:Optional[tuple[Optional['QuadTreeNode'], ...]]=None,
                 auto_split:bool=False):
        """Initialize the QuadTreeNode on the image frame.

        Args:
            value_matrix: The matrix containing the values of the node.
            ru_i: The right upper row index of the node.
            ru_j: The right upper column index of the node.
            max_depth: The maximum depth of the tree.
            id_sequence: The sequence of the node's id, including all its ancestors.
            neighbors: The identifiers of the node's neighbors, in the order of [left, right, top, bottom]. See Notes.
            auto_split: If the node should be split automatically.

        Notes:
            If the neighbors info is not provided (None), it can be updated later using the `update_neighbors` method.
            If one of the neighbors is not available, it can be set as None.
        """
        if not isinstance(id_sequence, tuple):
            raise ValueError(f"[{self.__class__.__name__}] id_sequence must be a tuple, got {type(id_sequence)}")
        if len(id_sequence) > max_depth:
            raise ValueError(f"[{self.__class__.__name__}] id_sequence length {len(id_sequence)} exceeds max_depth {max_depth}")
        self._max_depth = max_depth

        self._values = value_matrix
        self._ru_indices = (ru_i, ru_j)
        self._id_sequence = id_sequence
        self._neighbors = neighbors

        self._children : list[QuadTreeNode] = [] 
        if auto_split and self.state == TreeNodeState.MIX:
            self.split(max_depth, auto_split=auto_split)

    @classmethod
    def init_root(cls, value_matrix: np.ndarray, max_depth: int, auto_split:bool=False):
        """Initialize the root node."""
        return cls(value_matrix, 0, 0,
                   max_depth=max_depth,
                   id_sequence=(), # root node has no ancestors
                   neighbors=(None, None, None, None),
                   auto_split=auto_split)
    
    @property
    def state(self) -> TreeNodeState:
        """Get the state of the node. The state can be UNKNOWN, MIX, FULL, or FREE."""
        self._check_state(self._max_depth)
        return self._state

    @property
    def values(self) -> np.ndarray:
        """Get the value of the node."""
        return self._values
    
    @property
    def width(self) -> int:
        """Get the width of the node."""
        return self.values.shape[1]
    
    @property
    def height(self) -> int:
        """Get the height of the node."""
        return self.values.shape[0]
    
    @property
    def i(self) -> int:
        """Get the row index of the node's right upper corner."""
        return self._ru_indices[0]
    
    @property
    def j(self) -> int:
        """Get the column index of the node's right upper corner."""
        return self._ru_indices[1]
    
    @property
    def center_ij(self) -> tuple[int, int]:
        """Get the center (i, j) of the node."""
        return (self.i + self.height // 2, self.j + self.width // 2)
    
    @property
    def center(self) -> tuple[float, float]:
        """Get the center (x, y) of the node."""
        return (self.j + self.width // 2, self.i + self.height // 2)

    @property
    def vertices(self) -> list[tuple[int, int]]:
        """Get the vertices of the node. Note that i and j are swapped to get (x, y) coordinates."""
        return [
            (self.j, self.i),
            (self.j + self.width, self.i),
            (self.j + self.width, self.i + self.height),
            (self.j, self.i + self.height),
        ]

    @property
    def id_sequence(self) -> tuple[int, ...]:
        """Get the id sequence of the node."""
        return self._id_sequence

    @property
    def depth(self) -> int:
        """Get the depth of the node. The root node has a depth of 1."""
        return len(self._id_sequence)
    
    @property
    def children(self) -> list['QuadTreeNode']:
        """Get the children of the node."""
        return self._children

    @property
    def neighbors(self) -> Optional[tuple[Optional['QuadTreeNode'], ...]]:
        """Get the identifiers of the node's neighbors, in the order of [left, right, top, bottom]."""
        return self._neighbors

    @property
    def is_leaf(self) -> bool:
        """If the ndoe has no children, it is a leaf node."""
        return len(self._children) == 0

    def _check_state(self, max_depth: int):
        """Check the state of the node."""
        if np.all(self.values < 0.01):
            self._state = TreeNodeState.FREE
        elif len(self.id_sequence) == max_depth:
            self._state = TreeNodeState.FULL
        else:
            self._state = TreeNodeState.MIX

    def _check_node_type(self):
        """Check the type of the node."""
        raise NotImplementedError # TODO implement this method
        
    def _find_neighbors_for_child(self, i: int, j: int, num_splits: int):
        if self.neighbors is None:
            return None
        
        if i == 0:
            top_neighbor = self.neighbors[2]
        else:#i=1
            top_neighbor = self._children[(i-1)*num_splits + j]
        if i == num_splits - 1: #i=1
            bottom_neighbor = self.neighbors[3]
        else: #i=0
            bottom_neighbor = self._children[(i+1)*num_splits + j]

        if j == 0:
            left_neighbor = self.neighbors[0]
        else:#j=1
            left_neighbor = self._children[i*num_splits + j - 1]

        if j == num_splits - 1: #j=1
            right_neighbor = self.neighbors[1]
        else: #i=0
            right_neighbor = self._children[i*num_splits + j + 1]

        return (left_neighbor, right_neighbor, top_neighbor, bottom_neighbor)
    
    def update_neighbors(self, left: 'QuadTreeNode', right: 'QuadTreeNode', top: 'QuadTreeNode', bottom: 'QuadTreeNode'):
        """Update the neighbors of the node."""
        self._neighbors = (left, right, top, bottom)

    def split(self, max_depth: int, num_splits:int=2, auto_split:bool=False):
        """Split the node into sub-nodes.
        
        Args:
            max_depth: The maximum depth of the tree.
            num_splits: The number of splits in each dimension. Default is 2.
            auto_split: If the node should be split automatically.
            
        Notes:
            The order of the children for a 2x2 split is as follows:
            ```
            1 2
            3 4
            ```
        """
        new_height = self.height // num_splits
        new_width = self.width // num_splits
        cnt = 0
        for ii in range(num_splits):
            for jj in range(num_splits):
                child_grid = self.values[ii*new_height : (ii+1)*new_height, 
                                         jj*new_width  : (jj+1)*new_width]
                child_id_sequence = tuple(list(self._id_sequence) + [cnt])
                child_node = QuadTreeNode(
                    child_grid, self.i+ii*new_height, self.j+jj*new_width, 
                    max_depth=max_depth,
                    id_sequence=child_id_sequence,
                    neighbors=(None, None, None, None),
                    auto_split=auto_split
                )
                self._children.append(child_node)
                cnt += 1

        if self.neighbors is not None:
            for ii in range(num_splits):
                for jj in range(num_splits):
                    child_node = self._children[ii*num_splits + jj]
                    child_node.update_neighbors(*self._find_neighbors_for_child(ii, jj, num_splits))

 














