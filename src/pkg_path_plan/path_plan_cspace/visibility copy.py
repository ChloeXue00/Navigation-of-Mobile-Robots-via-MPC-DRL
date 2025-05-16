from extremitypathfinder.extremitypathfinder import PolygonEnvironment
from typing import Tuple, List

class VisibilityPathFinder:
    """
    Description:
    Generate the reference path via the visibility graph and A* algorithm.
    Attrs:
    env:The environment object of solving the visibility graph.
    Funcs:
    __prepare: Prepare the visibility graph including preprocess the map.
    get_ref_path: Get the (shortest) reference path.
    """
    def __init__(self, boundary_coords, obstacle_list, verbose = False):
        self.__prt_name = '[LocalPath-Visibility]'
        self.boundary_coords = boundary_coords
        self.obstacle_list = obstacle_list
        self.vb = verbose
        self.__prepare()
        
    def __prepare(self):
        self.env = PolygonEnvironment()
        self.env.store(self.boundary_coords, self.obstacle_list)
        self.env.prepare()
        
    def update_env(self):
        pass
    
    def get_ref_path(self, start_pos: tuple, end_pos: tuple) -> List[tuple]:
        """
        Description:
            generate the initially guessed path based on obstacles and boundaries specified during preparation

        Args:
            start_pos (tuple): the x,y coordinates
            end_pos (tuple): The x, y coordinates.

        Returns:
            List[tuple]: path: List of coordinates of initial path
        """
        if self.vb:
            print(f'{self.__prt_name} Reference path generated')
            
        try:
            path, dist = self.env.find_shortest_path(start_pos[:2],end_pos[:2])
        except Exception as e:
            print(f'{self.__prt_name} With start {start_pos} and goal {end_pos}.')
            print(f'{self.__prt_name} with boundary {self.boundary_coords}.')
            raise e
        return path
    