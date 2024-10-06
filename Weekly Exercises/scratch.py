import numpy as np
import pandas as pd

# pos = np.random.rand(3,3)
pos = np.array(((1,2,3), (4,5,6), (7,8,9)))
x = pos[:,0:1]
print(x)
print('----------------')
print(x.T)
print('----------------')
print(x.T-x)
print('----------------')
dx = np.zeros((3,3))
for i in range(3):
    # starts at i = 0 (first row)
    for j in range(3):
        # starts at j = 0 (first column)
        dx[i, j] = x[j, 0] - x[i, 0]

print(dx)


ax = np.array([1, 1, 1, 1])
ay = np.array([2, 2, 2, 2])
az = np.array([3, 3, 3, 3,])

print("ax:", ax)
print("ay:", ay)
print("az", az)

a = np.vstack((ax, ay, az))
print("a:", a)
print("a.T:", a.T)


fig, (ax1, ax2) = plt.subplots(1, 2, sharex = False, sharey = False) # subplots for scatter and pie chart
	initial, = ax1.plot(pos[:,0:1], pos[:,1:2], 'o', color='turquoise') # plot of initial snapshot


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


	""" ---------------------------------------------------------------------------------------------------------------- """
	"""                                                       Animation                                                  """
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
	anim = animation.FuncAnimation(fig=fig_anim, func=animate, frames=len(pos_time), interval=10)
	anim_2d = animation.FuncAnimation(fig=fig_2d, func=animate_2d, frames=len(pos_2d_time), interval=10)

	# Saving 2D and 3D animations
	anim.save('3d particle animation.gif', writer='pillow', fps=24)
	anim_2d.save('2d particle animation.gif', writer='pillow', fps=24)

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
