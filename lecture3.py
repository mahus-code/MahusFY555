import numpy as np
import pandas as pd

N = 1_000

# x = np.random.random(N)
# np.savetxt('my_array.txt', x) # not enough memory to allocate

'''
with open('my_array2.txt', 'w') as filehandle:
    for x in range(N):
        filehandle.write('%s\n' %np.random.random())
'''
'''
fmax = 0
with open('my_array2.txt', 'r') as f:
    for entry in f:
        if float(entry)>fmax:
            fmax = float(entry)
print(fmax)
'''

df = pd.DataFrame({ "Name":["Higgs", "electron", "tau neutrino", "muon"], "Mass": [125, 0.000511, 
                        None, 0.1], "charge":[0,-1,0,-1]})
print(df)

# print(df["charge"].max())
print(df.describe()) # gives mean, std, count, max, etc.

df.to_csv('particle_data.csv')

df2 = pd.read_csv('particle_data.csv')
#print(df2)

df["ratio"] = 100*(df["Mass"]/df["charge"])
#print(df)


for chunk in pd.read_csv('particle_data.csv', chunksize = 2): # gives us chunks of rows of the data
    # do stuff
    print(chunk, '\n\n\n')

df3 = pd.read_csv('particle_data.csv', usecols=["Mass"])
print(df3)

df4 = pd.read_csv('particle_data.csv', dtype={"charge": "int8"})