import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
import random as rand
#import pandas as pd
import psutil
import time


def check_for_nonZero(array: NDArray[np.float64]) -> None:
	count = 0
	flag = False
	print("ndim", np.ndim(array))
	if np.ndim(array) == 1:
		i = np.shape(array)
		print("here")
		for i in range(len(array)):
			if array[i] > 0:
				print("Non-zero detected:", array[i])
				flag = True
				count += 1
	elif np.ndim(array) == 2:
		i, j = np.shape(array)
		for n in range(i):
			for m in range(j):
				if array[n, m] > 0:
					print("Non-zero detected:", array[n, m])
					count += 1
	elif np.ndim(array) == 3:
		i, j, u = np.shape(array)
		for k in range(i):
			for n in range(j):
				for m in range(u):
					if array[k, n, m] > 0:
						print("Non-zero detected:", array[k, n, m])
						count += 1
	if not flag:
		print("No non-zeros")
	print("Non-zero Count:", count)
	return

def randomCompliment(list: str) -> None:
	i: int = rand.randint(0, len(list)-1)
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
	inv_r3 = (dx**2 + dy**2 + dz**2 + softening**2)
	inv_r3[inv_r3>0] = inv_r3[inv_r3>0]**(-1.5)

	ax = G * (dx * inv_r3) @ mass
	ay = G * (dy * inv_r3) @ mass
	az = G * (dz * inv_r3) @ mass

	# pack together the acceleration components
	a: NDArray[np.float64] = np.hstack((ax,ay,az))
	
	return a

def getAcc_slow( pos2: NDArray[np.float64], mass2: NDArray[np.float64], G2: float, softening2: float ) -> NDArray[np.float64]:
	x = pos2[:,0:1] # all rows, and 1 column (0), doesn't include 1
	y = pos2[:,1:2]
	z = pos2[:,2:3]
	i, j = np.shape(x)

	ax = np.zeros(i)
	ay = np.zeros(i)
	az = np.zeros(i)
	a = np.zeros((i, j))

	for n in range(i):
		for m in range(i):
			dx = x[m, 0].item() - x[n, 0].item()
			dy = y[m, 0].item() - y[n, 0].item()
			dz = z[m, 0].item() - z[n, 0].item()

			dist_squared = (dx**2 + dy**2 + dz**2 + softening2**2)
			if dist_squared > 0:
				inv_r3: float = (dist_squared)**(-1.5)

				ax[n] += G2 * (dx*inv_r3) * mass2[m, 0]
				ay[n] += G2 * (dy*inv_r3) * mass2[m, 0]
				az[n] += G2 * (dz*inv_r3) * mass2[m, 0]
	
	# pack together the acceleration components
	a = np.vstack((ax,ay,az)).T

	return a
			
""" N-body simulation """
	
# Simulation parameters
N         = 100    # Number of particles
t         = 0      # current time of the simulation
tEnd      = 10.0   # time at which simulation ends
dt        = 0.01   # timestep
softening = 0.1    # softening length
G         = 1.0    # Newton's Gravitational Constant
	
# Generate Initial Conditions
np.random.seed(17)            # set the random number generator seed
	
mass = 20.0*np.ones((N,1))/N  # total mass of particles is 20
pos = np.random.rand(N,3)*10 # randomly selected positions and velocities, normal dist from 0 to 1 - times by 10
pos_slow = np.random.rand(N,3)*10

vel  = np.random.randn(N,3)
vel_slow = np.random.randn(N,3)
# Convert to Center-of-Mass frame
vel -= np.mean(mass * vel,0) / np.mean(mass)
vel_slow -= np.mean(mass * vel,0) / np.mean(mass)
# calculate initial gravitational accelerations
acc = getAcc( pos, mass, G, softening )
acc_slow = getAcc_slow( pos_slow, mass, G, softening)	
# number of timesteps
Nt = int(np.ceil(tEnd/dt))
	
# save energies, particle orbits for plotting trails
pos_save = np.zeros((N,3,Nt+1))
pos_save[:,:,0] = pos
t_all = np.arange(Nt+1)*dt

compliments = ["You're strong", "You're good at programming", 
			   "You are handsome", "You are very intelligent", 
			   "You are extremely smart"]

# Simulation Main Loop


fig, (ax1, ax2) = plt.subplots(1, 2, sharex = False, sharey = False)
initial, = ax1.plot(pos[:,0:1], pos[:,1:2], 'o', color='turquoise')
start_time_quick = time.time()
for i in range(N):
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

end_time_quick = time.time()
final, = ax1.plot(pos[:,0:1], pos[:,1:2], 'o', color='teal')
ax1.set_title('The x-position of N particles versus the y-position')
ax1.grid(True)
ax1.set_xlabel('X-position')
ax1.set_ylabel('Y-position')
ax1.legend([initial, final], ['Initial', 'Final'], loc='upper left')

neg_count = 0
for i in range(N):
	for j in range(3):
		if pos[i,j] < 0:
			neg_count += 1

neg_percentage = neg_count/N * 100
pie_data = [neg_percentage, 100 - neg_percentage]
pie_labels = ['Percentage of particles \n with negative x,y and/or z positions', 'Particles with positive positions']

# Second subplot: pie chart
wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, colors=['mediumvioletred', 'midnightblue'], autopct='%1.1f%%', textprops={'size': 'smaller'}, radius=0.65)
for autotext in autotexts:
    autotext.set_color('white')
ax2.set_title('Percentage of Particles with Negative Positions')

'''----------------------------------------------------------------------------------------------'''

N         = 100    # Number of particles
t         = 0      # current time of the simulation
tEnd      = 10.0   # time at which simulation ends
dt        = 0.01   # timestep
softening = 0.1    # softening length
G         = 1.0    # Newton's Gravitational Constant

start_time_slow = time.time()
for i in range(N):
	vel += acc_slow * dt/2.0
	pos_slow += vel_slow * dt

	acc_slow = getAcc_slow( pos_slow, mass, G, softening )
	vel += acc * dt/2.0

	t += dt
end_time_slow = time.time()



print(f"Resident Memory Size: {psutil.Process().memory_info().rss / (1024 * 1024)} MB")
print(f"Virtual Memory Size: {psutil.Process().memory_info().vms / (1024 * 1024)} MB")
print(f"Swapped Memory Size: {psutil.Process().memory_info().vms / (1024 * 1024) - psutil.Process().memory_info().rss / (1024 * 1024)}")

print(f"Time for N-body process 1: {end_time_quick-start_time_quick:.2f} seconds")	
print(f"Time for N-body process 2: {end_time_slow-start_time_slow:.2f} seconds")

plt.show()