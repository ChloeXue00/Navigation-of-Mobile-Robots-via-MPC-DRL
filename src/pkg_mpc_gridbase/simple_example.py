
import numpy as np

'''
In this interface, you can try to get the reference states via the set_ref_states method.
Then, you can run the controller via the run_step method according to the reference states.
'''

class SimpleController:
    def __init__(self) -> None:
        pass

    def load_init_states(self, current_state: np.ndarray) -> None:
        '''This method is used to initialize the state.'''
        if (not isinstance(current_state, np.ndarray)):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}')
        self.state = current_state


    def set_ref_states(self, ref_states: np.ndarray) -> None:
        '''This method is used to set the reference states.'''
        self.ref_states = ref_states


    def check_termination_condition(self) -> bool:
        '''This method is used to check the termination condition.
        This method should return True if the controller is terminated.'''
        return False

    def run_step(self):
        '''This method is used to run one step of the controller.
        This method should return the control input (v, w).'''
        pass


    """from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np

points = np.random.randint(0, 10, size=(15, 2))  # Random points in 2-D

hull = ConvexHull(points)

fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 3))

for ax in (ax1, ax2):
    ax.plot(points[:, 0], points[:, 1], '.', color='k')
    if ax == ax1:
        ax.set_title('Given points')
    else:
        ax.set_title('Convex hull')
        for simplex in hull.simplices:
            ax.plot(points[simplex, 0], points[simplex, 1], 'c')
        ax.plot(points[hull.vertices, 0], points[hull.vertices, 1], 'o', mec='r', color='none', lw=1, markersize=10)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
plt.show()"""
