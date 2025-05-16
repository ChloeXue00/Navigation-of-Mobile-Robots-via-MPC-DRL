import numpy as np

class CostMonitor:
    def __init__(self) -> None:
        self._state = None

    def set_state(self, state: np.ndarray) -> None:
        self._state = state

    def cost_refpath_deviation(self, line_segments: np.ndarray, weight: float=1.0) -> float:
        """Reference deviation cost (weighted squared) penalizes on the deviation from the reference path.
        
        Arguments:
            line_segments: The (m*n)-dim var with m n-dim points.
        """
        distances_sqrt = []
        for i in range(line_segments.shape[0]-1):
            distance = self.dist_to_lineseg(self._state[:2], line_segments[i:i+2,:2])
            distances_sqrt.append(distance**2)
        cost = min(distances_sqrt) * weight
        return cost
    
    

    @staticmethod
    def dist_to_lineseg(point: np.ndarray, line_segment: np.ndarray):
        """Calculate the distance from a target point to a line segment.

        Arguments:
            point: The (1*n)-dim target point.
            line_segment: The (2*n) edge points.

        Returns:
            distance: The (1*1)-dim distance from the target point to the line segment.
        """
        (p, s1, s2) = (point[:2], line_segment[0,:], line_segment[1,:])
        s2s1 = s2-s1 # line segment
        t_hat = np.dot(p-s1,s2s1)/(s2s1[0]**2+s2s1[1]**2+1e-16)
        t_star = min(max(t_hat,0.0),1.0) # limit t
        temp_vec = s1 + t_star*s2s1 - p # vector pointing to closest point
        distance = np.linalg.norm(temp_vec)
        return distance
    

if __name__ == "__main__":
    state = np.array([0.0, 0.0, 0.0])
    line_segments = np.array([[0.0, 1.0, 0.0],
                              [1.0, 0.0, 0.0],
                              [1.0, 1.0, 0.0]])

    cm = CostMonitor()
    cm.set_state(state)
    cost_refpath_deviation = cm.cost_refpath_deviation(line_segments)

    print(f"cost_refpath_deviation: {cost_refpath_deviation}")
