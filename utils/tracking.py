import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

class EnvironmentCapture():
    # TODO: Add wandb integration
    def __init__(self):
        self.captures = {}

    def add_step(self, key, value):
        if key in self.captures.keys():
            self.captures[key].append(value)
        else:
            self.captures[key] = [value]

    def stack(self):
        stacks = {}
        for key in self.captures:
            stacks[key] = np.array(self.captures[key])
        return stacks

def create_animation(array, save_path=None, fps=60):
    ansi_colors = [
    '#000000', '#0000aa', '#00aa00', '#00aaaa', 
    '#aa0000', '#aa00aa', '#ff5500', '#ffdd00', 
    '#00ffaa', '#aa0000'  # Add more if needed, or customize as per ANSI scheme
    ]
    N, width, height = array.shape
    fig, ax = plt.subplots()
    cmap = mcolors.ListedColormap(ansi_colors)
    im = ax.imshow(array[0], cmap=cmap, animated=True)
    ax.axis('off')  # Hide axis for a clean look
    def update(frame):
        im.set_array(array[frame])
        return [im]
    anim = FuncAnimation(fig, update, frames=N, interval=1000/fps, blit=True)   
    if save_path:
        writer = FFMpegWriter(fps=fps)
        anim.save(save_path, writer=writer)
    

    return anim
