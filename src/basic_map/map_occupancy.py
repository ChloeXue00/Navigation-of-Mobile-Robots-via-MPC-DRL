import numpy as np
from PIL import Image # type: ignore
from matplotlib.axes import Axes # type: ignore

from map_tools import blob_bounding


PathNode = tuple[float, float]


class OccupancyMap:
    """With image/matrix."""
    def __init__(self, map_image: np.ndarray, occupancy_threshold:int=120):
        map_image = self.__input_validation(map_image)
        self._width = map_image.shape[1]
        self._height = map_image.shape[0]

        self.__background = map_image
        self.__grayground = np.array(Image.fromarray(map_image).convert('L')) if map_image.shape[2]==3 else map_image[:,:,0]
        self.__binyground = (self.__grayground>occupancy_threshold)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __input_validation(self, map_image):
        if not isinstance(map_image, np.ndarray):
            raise TypeError('A map image must be a numpy array.')
        if len(map_image.shape) == 2: # add channel dimension
            map_image = map_image[:, :, np.newaxis]
        if len(map_image.shape) != 3:
            raise TypeError(f'A map image must have 2 or 3 dimensions; Got {len(map_image.shape)}.')
        if map_image.shape[2]==4: # the 4th channel will be discarded
            map_image = map_image[:, :, :3]
        if map_image.shape[2] not in [1, 3]: # the 4th channel can be alpha-channel
            raise TypeError(f'A map image must have 1/3/4 channels; Got {map_image.shape[2]}.')
        return map_image

    def __call__(self, binary_scale=False, gray_scale=True) -> np.ndarray:
        if binary_scale:
            return self.__binyground
        if gray_scale:
            return self.__grayground
        return self.__background
    
    @classmethod
    def from_image(cls, map_file_path: str):
        img = Image.open(map_file_path)
        map_data = np.array(img.convert('L'))
        return cls(map_data)

    def get_geometric_map(self, bounding_degree=4) -> tuple[list[PathNode], list[list[PathNode]]]:
        boundary_coords = [(0.0, 0.0), 
                           (0.0, float(self.height)), 
                           (float(self.width), float(self.height)), 
                           (float(self.width), 0.0)]
        obstacle_list = []
        blob_detector = blob_bounding.BlobBounding(bounding_degree)
        obstacle_list = blob_detector.get_bounding_polygons(self.__grayground)
        
        for coords in obstacle_list[::-1]:
            x1_left  = min(coords[:,0])
            x1_right = max(coords[:,0])
            y1_low = min(coords[:,1])
            y1_up  = max(coords[:,1])
            for other_coords in obstacle_list:
                sorted_coords_x = np.sort(other_coords[:,0])
                sorted_coords_y = np.sort(other_coords[:,1])
                x2_left  = sorted_coords_x[1]
                x2_right = sorted_coords_x[2]
                y2_low = sorted_coords_y[1]
                y2_up  = sorted_coords_y[2]
                if (x1_left>x2_left) & (x1_right<x2_right) & (y1_low>y2_low) & (y1_up<y2_up):
                    obstacle_list.pop(-1)
                    continue
        obstacle_list = [x.tolist() for x in obstacle_list]
        
        return boundary_coords, obstacle_list # type: ignore

    def plot(self, ax: Axes, binary_scale=False, gray_scale=True, **kwargs):
        ax.imshow(self(binary_scale, gray_scale), **kwargs)


if __name__ == '__main__':
    import matplotlib.pyplot as plt # type: ignore

    # Create an instance of OccupancyMap from an image file
    occupancy_map = OccupancyMap.from_image('/home/osama/aoara2024sp_gridbasedmpc/src/basic_map/map.png')

    # Print the occupancy_map object
    print(occupancy_map)

    # Create a figure with two subplots arranged vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

    # Plot the occupancy map in the first subplot
    occupancy_map.plot(ax1, cmap='gray')
    ax1.set_title('Occupancy Map')
    ax1.axis('off')

    # Get the geometric map coordinates
    boundary_coords, obstacle_list = occupancy_map.get_geometric_map()

    # Plot the occupancy map with geometric annotations in the second subplot
    occupancy_map.plot(ax2, cmap='gray')
    ax2.plot([x[0] for x in boundary_coords], [x[1] for x in boundary_coords], 'r', linewidth=3)
    for coords in obstacle_list:
        ax2.plot([x[0] for x in coords], [x[1] for x in coords], 'b', linewidth=3)
    ax2.set_title('Occupancy Map with Geometric Annotations')
    ax2.axis('off')

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Display the plots
    plt.show()
