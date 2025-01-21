from simulator import Grid
import moviepy.editor as mpy
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Example usage
grid = Grid(size=250, probs=[0.25, 0.25, 0.25, 0.25])
os.makedirs('assets/tmp', exist_ok=True)
for i in range(5000):
    grid.display_grid(seconds=0.1, title='Local Neighborhood', stamp=i, silent=True)
    grid.simulate(num_iters=3000, local=True)

# Create a GIF from the images
imgs = glob.glob('assets/tmp/*.png')
clip = mpy.ImageSequenceClip(imgs, fps=60)
clip.write_gif('assets/local_neighborhood.gif')
os.system('rm -rf assets/tmp')


