import numpy as np
from matplotlib import pyplot as plt # import matpblotlib.pyplot. as plt
from matplotlib import animation

# Set up figure:
fig = plt.figure()
ax = plt.axes(xlim = (0,2), ylim=(-2,2))
line, = ax.plot([], [], lw = 2)

# set up function for animation

def animate(frame):
    x = np.linspace(0,2,100)
    y = np.sin(2*np.pi*(x-0.01*frame))
    line.set_data(x,y)
    return line,

# animation:

anim = animation.FuncAnimation(fig, animate, frames = 100, interval = 50 , blit = False)
anim.save("sine.gif", writer = animation.PillowWriter(fps = 2))

plt.show()