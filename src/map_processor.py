from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from basic_map.map_geometric import GeometricMap
from basic_obstacle.geometry_plain import PlainPolygon
from skimage.morphology import erosion, square

class MapProcessor:
    """
    Class for processing maps, including inflation of obstacles.
    """

    def __init__(self, rescale_geo_occupancy: int = 100):
        """
        Initialize the MapProcessor.

        Args:
            rescale_geo_occupancy (int): Factor to convert grid coordinates to occupancy map coordinates.
        """
        self._current_map: Optional[np.ndarray] = None
        self._inflated_map: np.ndarray #occupancy grid
        self._rescale_geo_occupancy: int = rescale_geo_occupancy
        self.size_geo_map = 10  # needs to be changed to be more arbitrary? third element of geometric map_ boundary

    def load_map(self, geometry_map: GeometricMap, inflation_margin: Optional[float] = None) -> None:
        """
        Load a map with boundary coordinates and obstacle list, optionally inflating it.

        Args:
            boundary_coords (list): List of boundary coordinates.
            obstacle_list (list): List of obstacle coordinates.
            inflation_margin (float, optional): Margin for inflating obstacles.
        """
        # self._current_map = GeometricMap.from_raw(boundary_coords, obstacle_list)
        self._current_map = geometry_map.get_occupancy_map()
        if self._current_map is not None:
            if inflation_margin is not None:
                self._inflated_map = self.inflate_map(geometry_map, inflation_margin)
            else:
                self._inflated_map = self._current_map

    def inflate_map(self, original_map: GeometricMap, inflation_margin: float) -> np.ndarray:
        boundary_coords, obstacle_coords_list = original_map()

        # Inflate the boundaries
        boundary_coords = self._inflate_boundaries(boundary_coords, inflation_margin)
        geo_inflated_boundary_map = GeometricMap.from_raw(boundary_coords, obstacle_coords_list)
        self.geo_max_height = geo_inflated_boundary_map.get_boundary_scope()[3]
        occupancy_map_inflated_boundary = np.array(geo_inflated_boundary_map.get_occupancy_map())

        # Calculate the size of the structuring element based on the inflation margin
        structuring_element_size = max(1, int(inflation_margin * self._rescale_geo_occupancy))

        # Convert the 3D occupancy map array to a 2D binary array
        occupancy_map_binary = np.any(occupancy_map_inflated_boundary, axis=2)

        # Apply erosion to expand the unoccupied regions using morphology
        inflated_map = erosion(occupancy_map_binary, square(structuring_element_size))
        
        # Create a 3D RGB occupancy map
        rgb_map = np.zeros(inflated_map.shape + (3,), dtype=np.uint8)
        rgb_map[inflated_map == True] = [255, 255, 255]  # Dark cells (False) have RGB value [0, 0, 0]
        
        return rgb_map

    def _inflate_boundaries(self, boundary_coords: list, inflation_margin: float) -> list:
        """
        Inflate the boundaries of the map.

        Args:
            boundary_coords (list): List of boundary coordinates.
            inflation_margin (float): The margin for inflating the boundaries.

        Returns:
            list: Inflated boundary coordinates.
        """
        boundary_polygon = PlainPolygon.from_list_of_tuples(boundary_coords).inflate(-inflation_margin)
        return boundary_polygon()


def main():
    """
    when running this program, the plots (the obstacles) might look to be flipped (in Y),
    this has to do that what is being plotted is the data straight from get_occupancy_map
    which flips the elements in y. So say the original cell is (2,2) and the size of the map is 10
    then the cell will be shown as (2,8)

    Observe that the previous analogy is based on geometric_map coordinates in occupancy grid
    you have to rescale these elements by a rescale factor but the idea holds.
    """

    # Test data
    robot_width = 0.5
    boundary = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    """obstacle_list = [{"id_": 1, "vertices": [(2, 2), (3, 2), (3, 3), (2, 3)]},
                     {"id_": 2, "vertices": [(5, 3), (6, 3), (6, 6), (5, 6)]}
                     ]"""
    obstacle_list = [{"id_": 1, "vertices": [(1,1), (2,1), (2,2), (1,2)]}, 
                    {"id_": 2, "vertices": [(3,3), (4,3), (4,4), (3,4)]}
                    ]
    obstacle_info_list = [GeometricMap.dict_to_obstacle_info(obs) for obs in obstacle_list]
    geometric_map = GeometricMap(boundary, obstacle_info_list)
    original_map = np.array(geometric_map.get_occupancy_map())

    inflation_margin = robot_width

    # Instantiate MapProcessor
    processor = MapProcessor()

    # Load and inflate map
    processor.load_map(geometric_map, inflation_margin)
    inflated_map = processor._inflated_map
    print(inflated_map)
    # Visualize original and inflated maps
    def plot_map(map_data, title):
        plt.imshow(map_data)
        plt.title(title)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    # original_map = GeometricMap.from_raw(boundary_coords, obstacle_list).get_occupancy_map()
    plot_map(original_map, 'Original Map')
    plot_map(inflated_map, 'Inflated Map')


if __name__ == "__main__":
    main()
