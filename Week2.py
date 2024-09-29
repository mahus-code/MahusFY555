import numpy as np
import matplotlib.pyplot as plt
import random as rand

"""
Create Your Own N-body Simulation (With Python)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

def randomCompliment(list):
    i = rand.randint(0, len(list)-1)
    print("Your random compliment is: %s" % list[i])

def getAcc( pos, mass, G, softening ):
	"""
    Calculate the acceleration on each particle due to Newton's Law 
	pos  is an N x 3 matrix of positions
	mass is an N x 1 vector of masses
	G is Newton's Gravitational constant
	softening is the softening length
	a is N x 3 matrix of accelerations
	"""
	# positions r = [x,y,z] for all particles
	x = pos[:,0:1] # all rows, and 1 column (0), doesn't include 1
	y = pos[:,1:2]
	z = pos[:,2:3]

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
	a = np.hstack((ax,ay,az))

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
vel  = np.random.randn(N,3)
	
# Convert to Center-of-Mass frame
vel -= np.mean(mass * vel,0) / np.mean(mass)

# calculate initial gravitational accelerations
acc = getAcc( pos, mass, G, softening )
	
# number of timesteps
Nt = int(np.ceil(tEnd/dt))
	
# save energies, particle orbits for plotting trails
pos_save = np.zeros((N,3,Nt+1))
pos_save[:,:,0] = pos
t_all = np.arange(Nt+1)*dt

compliments = ["You're strong", "You're good at programming", "You are handsome", "You are very intelligent", "You are extremely smart"]

# Simulation Main Loop
for i in range(Nt):
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
				
	# save energies, positions for plotting trail
	pos_save[:,:,i+1] = pos

randomCompliment(compliments)


myArray = np.array([0, 2, 3])
print("My array is", myArray, "and has shape:", np.shape(myArray), "length:", len(myArray)) # (3,) means 1-D array with 3 elements

myArray2 = np.array([[0, 2, 3]])
print(myArray2[0])
print("My array2 is", myArray2, "and has shape:", np.shape(myArray2), "length:", len(myArray2), "dim:", np.ndim(myArray2)) #

myArray2D = np.array([ [0, 2, 3], [2, 1, 4] ])
print("My 2D array is", myArray2D, "and has shape:", np.shape(myArray2D), "length:", len(myArray2D), "dim:", np.ndim(myArray2D)) # (2, 3) means 2D matrix with 2 rows and 3 columns

myArray3D = np.array([ [ [1, 2, 3], [4, 5, 6] ], [[0, 0, 0], [5, 5, 5]] ])



def check_for_nonZero(array):
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
	return 0
			
check_for_nonZero(myArray)
check_for_nonZero(myArray2D)
check_for_nonZero(myArray3D)
