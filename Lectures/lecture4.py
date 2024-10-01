import matplotlib.pyplot as plt
import numpy as np

def my_func(x):
    return np.sin(x)*x

x = np.linspace(0,10,num=100)

from matplotlib import rc

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']}) # Default latex style
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text commandparams = {'legend.fontsize': 14,'legend.handlelength': 2,'axes.labelsize' : 18}
#plot.rcParams.update(params)plt.rcParams.update({'font.size': 14})
plt.rcParams['mathtext.fontset'] = 'stix'

plt.rc('xtick', labelsize = 6)
plt.rc('ytick',labelsize = 16)
plt.rc('legend', fontsize=18)

plt.plot(x,my_func(x),label='my function')
plt.ylabel('$\sin(x)\cdot x$')
plt.xlabel('$x$', fontsize = 18)
plt.legend(loc='upper left')
plt.title('Cosine function', fontsize = 18)
plt.show()