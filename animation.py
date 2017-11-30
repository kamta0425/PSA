"""
Created on Mon Oct 23 13:25:59 2017

@author: KAMMO, TANNY, AJ, SUSHI, GOWTHAMI
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import PSAreadingheader as psrd

"""  Minor 
"""
infile='/PSA/stage1_aps/00360f79fd6e02781457eda48f85da90.aps'
matplotlib.rc('animation', html='html5')
def plot_image(infile):
    data = psrd.read_data(infile)
    fig = matplotlib.pyplot.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    def animate(i):
        im = ax.imshow(np.flipud(data[:,:,7].transpose()), cmap = 'viridis')
        return [im]
    return matplotlib.animation.FuncAnimation(fig, animate, frames=range(0,data.shape[2]), interval=200, blit=True)

plot_image(infile)
"""  Matplotlib Animation 
"""

fig, ax = plt.subplots()

x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))


def animate(i):
    line.set_ydata(np.sin(x + i/10.0))  # update the data
    return line,


# Init only required for blitting to give a clean slate.
def init():
    line.set_ydata(np.ma.array(x, mask=True))
    return line,

ani = animation.FuncAnimation(fig, animate, np.arange(1, 200), init_func=init,
                              interval=25, blit=True)
plt.show()