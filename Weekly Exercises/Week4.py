import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import random as rand
import psutil
import time
from matplotlib import animation
import math

# Function to print a random compliment
def randomCompliment(list: str) -> None:
	i: int = rand.randint(0, len(list)-1) # indexing goes 0 -> 4 but len returns 5
	print("Your random compliment is: %s" % list[i]) 

def getAcc( pos: NDArray[np.float64], mass: NDArray[np.float64], G: float, softening: float ) -> NDArray[np.float64]:
	"""
	Calculate the acceleration on each particle due to Newton's Law 
	pos  is an N x 3 matrix of positions
	mass is an N x 1 vector of masses
	G is Newton's Gravitational constant
	softening is the softening length
	a is N x 3 matrix of accelerations
	"""
	# positions r = [x,y,z] for all particles
	x: NDArray[np.float64] = pos[:,0:1] # all rows, and 1 column (0), doesn't include 1
	y: NDArray[np.float64] = pos[:,1:2]
	z: NDArray[np.float64] = pos[:,2:3]

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x.T - x
	dy = y.T - y
	dz = z.T - z

	# matrix that stores 1/r^3 for all particle pairwise particle separations 
	inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2) # ** is raising to the power of
	inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

	ax = G * (dx * inv_r3) @ mass # * is elementwise multiplication
	ay = G * (dy * inv_r3) @ mass # @ is regular matrix multiplication (NxN) @ (Nx1) = (Nx1)
	az = G * (dz * inv_r3) @ mass

	# pack together the acceleration components
	a: NDArray[np.float64] = np.hstack((ax,ay,az))
	
	return a

def getAcc_slow( pos2: NDArray[np.float64], mass2: NDArray[np.float64], G2: float, softening2: float ) -> NDArray[np.float64]:
	x = pos2[:,0:1] # all rows, and 1 column (0), doesn't include 1
	y = pos2[:,1:2]
	z = pos2[:,2:3]

	i, j = np.shape(x) # returns rows and columns in i and j

	# Initialize empty arrays (Nx1) to store component accelerations
	ax = np.zeros(i)
	ay = np.zeros(i)
	az = np.zeros(i)

	# Array to store all the accelerations (Nx3)
	a = np.zeros((i, j))

	for n in range(i): 								# Loop through all rows of x
		for m in range(i):							# For each value of x, subtract all values of x
			dx = x[m, 0].item() - x[n, 0].item()	# we want x_j - x_i, .item() ensures we have a scalar
			dy = y[m, 0].item() - y[n, 0].item()
			dz = z[m, 0].item() - z[n, 0].item()

			dist_squared = (dx**2 + dy**2 + dz**2 + softening2**2) # we calculate r^2 for each seperation
			if dist_squared > 0: # ensure that r^2 is non-zero (positve)
				inv_r3: float = (dist_squared)**(-1.5) # compute 1/r^3 (since r = sqrt(dist_squared) )

				ax[n] += G2 * (dx*inv_r3) * mass2[m, 0] # adds the acceleration from the dx-contribution (so we go from NxN to 1xN)  
				ay[n] += G2 * (dy*inv_r3) * mass2[m, 0]
				az[n] += G2 * (dz*inv_r3) * mass2[m, 0]
	
	# pack together the acceleration components
	a = np.vstack((ax,ay,az)).T # ax, ay, az are (1xN) arrays. We pack them (3xN) --> transpose to (Nx3)

	return a
			
""" N-body simulation """
programStartTime= time.time()
# Simulation parameters
N         = 100    # Number of particles
t         = 0      # current time of the simulation
tEnd      = 10.0   # time at which simulation ends
dt        = 0.01   # timestep
softening = 0.1    # softening length
G         = 10.0    # Newton's Gravitational Constant
	
# Generate Initial Conditions
np.random.seed(17)            # set the random number generator seed
	
mass = 20.0*np.ones((N,1))/N  # total mass of particles is 20
pos = np.random.rand(N,3)*10 # randomly selected positions and velocities, normal dist from 0 to 1 - times by 10
pos_slow = np.random.rand(N,3)*10

vel  = np.random.randn(N,3)
vel_slow = np.random.randn(N,3)

# Convert to Center-of-Mass frame
vel -= np.mean(mass * vel, 0) / np.mean(mass) # ,0 means the mean is calculated across rows
vel_slow -= np.mean(mass * vel,0) / np.mean(mass)

# Calculate initial gravitational accelerations
acc = getAcc( pos, mass, G, softening )
acc_slow = getAcc_slow( pos_slow, mass, G, softening)	

# Number of timesteps
Nt = int(np.ceil(tEnd/dt))

compliments = ["You're strong", "You're good at programming", 
			   "You are handsome", "You are very intelligent", 
			   "You are extremely smart"]

# Simulation Main Loop

pos_time = [] # list to store the position at each time step
pos_2d_time = []
fig, (ax1, ax2) = plt.subplots(1, 2, sharex = False, sharey = False) # subplots for scatter and pie chart
initial, = ax1.plot(pos[:,0:1], pos[:,1:2], 'o', color='turquoise') # plot of initial snapshot

fig_anim = plt.figure() # setup figure for the 3D animation
plot = fig_anim.add_subplot(projection='3d')
scatter = plot.scatter([pos[:,0]], [pos[:,1]], [pos[:,2]]) # Initial position needs to be given - otherwise blank figure

# Add axis labels + title for 3D animation
plot.set_xlabel('X-position')
plot.set_ylabel('Y-position')
plot.set_zlabel('Z-position')
plot.set_title('3D animation of N-body simulation')

# Setup 2d animation
fig_2d = plt.figure()
ax_2d = plt.axes()
line_2d = ax_2d.scatter([pos[:,0]], [pos[:,1]], zorder=2) # Initial positions are given, zorder ensures points above grid

# Lists to store mean x, time and std
mean_x = []
time_x = []
std_x = []

start_time_quick = time.time() # timer starts for the quick simulation
for i in range(Nt):
	# save time with position
	pos_time.append(pos.copy()) # .copy() avoids referencing
	pos_2d_time.append(pos[:,0:2].copy()) # only want x,y columns

	mean_x.append(np.mean(pos[:,0]))
	time_x.append(t)
	std_x.append(np.std(pos[:,0]))

	# (1/2) kick
	vel += acc * dt/2.0
	
	# drift
	pos += vel * dt
		
	# update accelerations
	acc = getAcc( pos, mass, G, softening )

	# (1/2) kick
	vel += acc * dt/2.0
		
	# update time
	t += dt

end_time_quick = time.time() # stop timer for the quick simulation

# Plot the final snapshot of the simulation
final, = ax1.plot(pos[:,0:1], pos[:,1:2], 'o', color='teal')

# Add labels and title to 2d scatter
ax1.set_title('The x-position of N particles versus the y-position')
ax1.grid(True)
ax1.set_xlabel('X-position')
ax1.set_ylabel('Y-position')
ax1.legend([initial, final], ['Initial', 'Final'], loc='upper left')

# Find the amount of particles with a negative position component
neg_count = 0
for i in range(N):
	for j in range(3):
		if pos[i,j] < 0:
			neg_count += 1

# Calculate the percentage of negative
neg_percentage = neg_count/N * 100
pie_data = [neg_percentage, 100 - neg_percentage]
pie_labels = ['Percentage of particles \n with negative x,y and/or z positions', 'Particles with positive positions']

# Autopct adds the percentage to the piechart
wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, colors=['mediumvioletred', 'midnightblue'], 
								   autopct='%1.1f%%', textprops={'size': 'smaller'}, radius=0.65)

# Set the generated text to white (percentage text)
for autotext in autotexts:
    autotext.set_color('white')
# Title of pie chart
ax2.set_title('Percentage of Particles with Negative Positions')

# Plotting the mean of the x-coordinate versus time
# with standard deviation shown as errorbars
fig_mean, ax_mean = plt.subplots()
plt.errorbar(time_x, mean_x, yerr=std_x)
ax_mean.set_xlabel('Time (s)')
ax_mean.set_ylabel('Mean x coordinate')
ax_mean.set_title('Time (s) versus mean x coordinate of a N-body simulation')
ax_mean.grid(True)

""" ---------------------------------------------------------------------------------------------------------------- """
"""                                                       Animation                                                  """


# Add axis labels + title + grid for 2D animation
ax_2d.set_xlabel('X-position')
ax_2d.set_ylabel('Y-position')
ax_2d.set_title('2D animation of N-body simulation')
ax_2d.grid(True, zorder=1) # lower zorder, grid at back

# 3D Animate function: uses the list containing the position arrays for a given time step
def animate(frame):
	scatter._offsets3d = (pos_time[frame][:,0], pos_time[frame][:,1], pos_time[frame][:,2]) # _offsets3d when using scatter 3d
	return scatter

# 2D Animate function
def animate_2d(frame):
    line_2d.set_offsets(pos_2d_time[frame]) # _offsets for 2d scatter
    return line_2d

# Animations
anim = animation.FuncAnimation(fig=fig_anim, func=animate, frames=len(pos_time), interval=50)
anim_2d = animation.FuncAnimation(fig=fig_2d, func=animate_2d, frames=len(pos_2d_time), interval=50)

# Saving 2D and 3D animations
#anim.save('3d particle animation.gif', writer='pillow', fps=24)
#anim_2d.save('2d particle animation.gif', writer='pillow', fps=24)


'''----------------------------------------------------------------------------------------------'''
'''                      Slow Simulation                                                         '''

N         = 100    # Number of particles
t         = 0      # current time of the simulation
tEnd      = 10.0   # time at which simulation ends
dt        = 0.01   # timestep
softening = 0.1    # softening length
G         = 1.0    # Newton's Gravitational Constant

# Timer starts for the slow simulation
start_time_slow = time.time()
for i in range(Nt):
	vel += acc_slow * dt/2.0
	pos_slow += vel_slow * dt

	acc_slow = getAcc_slow( pos_slow, mass, G, softening )
	vel += acc_slow * dt/2.0

	t += dt
end_time_slow = time.time() # timer ends for the slow simulation


# Print the memory usuage of the program
print(f"Resident Memory Size: {psutil.Process().memory_info().rss / (1024 * 1024)} MB") # RAM used
print(f"Virtual Memory Size: {psutil.Process().memory_info().vms / (1024 * 1024)} MB") # Total memory
print(f"Swapped Memory Size: {psutil.Process().memory_info().vms / (1024 * 1024) - psutil.Process().memory_info().rss / (1024 * 1024)} MB") # Swapped memory for the script

# Print the time for the quick and slow simulation
print(f"Time for N-body process 1: {end_time_quick-start_time_quick:.2f} seconds")	
print(f"Time for N-body process 2: {end_time_slow-start_time_slow:.2f} seconds")

# Print a random compliment
randomCompliment(compliments)
programEndTime = time.time()
print("Program Run time =", programEndTime-programStartTime)

plt.show()