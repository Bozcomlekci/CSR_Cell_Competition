import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import time

class Grid:
    def __init__(self, size=250, probs=None):
        """
        Initialize the grid with a given size and state probabilities.
        
        Args:
            size (int): Size of the grid (size x size).
            probs (list): Probabilities for initial states [C, S, R, W].
        """
        self.size = size
        self.state_dict = {0: 'C', 1: 'S', 2: 'R', 3: 'W'}
        self.probs = probs if probs else [0.25, 0.25, 0.25, 0.25]
        self.grid = self.initialize_grid()

    def initialize_grid(self):
        """Initialize the grid with states based on probabilities."""
        nums = list(self.state_dict.keys())
        return np.array([[random.choices(nums, weights=self.probs)[0] for _ in range(self.size)] for _ in range(self.size)])

    def display_grid(self, seconds=0.1, title='CSR Cell Automata', stamp='init', silent=False):
        """Display the current state of the grid."""
        plt.imshow(self.grid, interpolation='none', cmap=plt.get_cmap('Set1'))
        plt.title(title)
        cbar = plt.colorbar()
        cbar.set_ticks(range(len(self.state_dict)))
        cbar.set_ticklabels(list(self.state_dict.values()))
        plt.clim(0, len(self.state_dict) - 1)
        if not silent:
            plt.draw()
            plt.pause(seconds)
        # save the current figure
        if stamp:
            plt.savefig(f'assets/tmp/{stamp}.png')
        plt.clf()
        
    

    def random_point(self):
        """Pick a random point in the grid."""
        return random.choices(range(self.size), k=2)

    def simulate(self, num_iters, local=True, delta_c=1/3, delta_r=10/32, T=3/4, delta_s_0=1/4):
        """
        Run the simulation for a given number of iterations.

        Args:
            num_iters (int): Number of iterations.
            local (bool): Whether to use local or global neighborhood.
            delta_c (float): Death probability for C.
            delta_r (float): Death probability for R.
            T (float): Toxicity parameter for S.
            delta_s_0 (float): Base death probability for S.
        """
        for _ in range(num_iters):
            death_prob = random.random()
            p = self.random_point()
            curr_p = self.grid[p[0], p[1]]

            if local:
                # Local neighborhood (8 cells)
                f_c = f_s = f_r = 0
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if not (i == 0 and j == 0):
                            nb = self.grid[(p[0] + i) % self.size, (p[1] + j) % self.size]
                            if nb == 0: f_c += 1
                            if nb == 1: f_s += 1
                            if nb == 2: f_r += 1
                f = [f_c / 8, f_s / 8, f_r / 8]
            else:
                # Global neighborhood
                f_c = np.count_nonzero(self.grid == 0) - (curr_p == 0)
                f_s = np.count_nonzero(self.grid == 1) - (curr_p == 1)
                f_r = np.count_nonzero(self.grid == 2) - (curr_p == 2)
                total_cells = self.size**2 - 1
                f = [f_c / total_cells, f_s / total_cells, f_r / total_cells]

            delta_s = delta_s_0 + T * f[0]

            if curr_p == 3 and death_prob < f[0]:
                self.grid[p[0], p[1]] = 0
            elif curr_p == 3 and death_prob < f[0] + f[1]:
                self.grid[p[0], p[1]] = 1
            elif curr_p == 3 and death_prob < f[0] + f[1] + f[2]:
                self.grid[p[0], p[1]] = 2
            elif curr_p == 3:
                self.grid[p[0], p[1]] = 3

            if curr_p == 0 and death_prob < delta_c:
                self.grid[p[0], p[1]] = 3
            elif curr_p == 1 and death_prob < delta_s:
                self.grid[p[0], p[1]] = 3
            elif curr_p == 2 and death_prob < delta_r:
                self.grid[p[0], p[1]] = 3

