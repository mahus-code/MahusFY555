import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from astroML.datasets import sdss_corrected_spectra

# Fetch the data
data = sdss_corrected_spectra.fetch_sdss_corrected_spectra()
spectra = sdss_corrected_spectra.reconstruct_spectra(data)
lam = sdss_corrected_spectra.compute_wavelengths(data)
#print(np.unique(data))

# First we normalize the spectra (row wise)
print(spectra.shape)
normalized_spectra = preprocessing.normalize(spectra, axis=1)

# Calculate the mean and standard deviation across all spectra
mean_flux = normalized_spectra.mean(axis=0)
std_flux = normalized_spectra.std(axis=0)

# Utilize the mean spectrum code from the example
wavelengths = lam
plt.plot(lam, mean_flux, color='black')
plt.fill_between(wavelengths, mean_flux - std_flux, mean_flux + std_flux, color='#CCCCCC')
plt.xlim(wavelengths[0], wavelengths[-1]) # Gives last element
plt.ylim(0, 0.06)
plt.xlabel('wavelength (Angstroms)')
plt.ylabel('scaled flux')
plt.title('Mean Spectrum')
plt.grid(True)
plt.show()

X = normalized_spectra
from sklearn.decomposition import PCA

# Perform PCA
rpca = PCA(n_components=4, svd_solver='randomized', random_state=0)
X_proj = rpca.fit_transform(X)
y = data['z'] # where z is the redshift

plt.figure()
scatter =plt.scatter(X_proj[:, 0], X_proj[:, 1], c=y, s=4, lw=0, vmin=y.min(), vmax=y.max(), cmap=plt.cm.jet)
plt.colorbar(scatter, label="Redshift (z)")
plt.xlabel('coefficient 1')
plt.ylabel('coefficient 2')
plt.title('PCA projection of Spectra')
plt.grid(True)
plt.show()

plt.figure()
l = plt.plot(wavelengths, rpca.mean_ - 0.15)
c = l[0].get_color()
plt.text(7000, -0.16, "mean", color=c)
for i in range(4):
    l = plt.plot(wavelengths, rpca.components_[i] + 0.15 * i)
    c = l[0].get_color()
    plt.text(7000, -0.01 + 0.15 * i, "component %i" % (i + 1), color=c)
plt.ylim(-0.2, 0.6)
plt.xlabel('wavelength (Angstroms)')
plt.ylabel('scaled flux + offset')
plt.title('Mean Spectrum and Eigen-spectra')
plt.grid(True)
plt.show()

print("This script loads SDSS data from astroML, normalizes the spectra, and plots the mean spectrum.") 
print("A PCA analysis is performed to reduce the dimensionality of the data to four components.")
print("The first scatter plot of the second component versus the first component illustrates how the spectra vary along these two axes, which capture the most variance and thereby provide the most information about the data.")
print("The color-coding of the plot shows how redshift correlates with these two components.")
print("The final plot shows the four principal components (eigen-spectra). The first component accounts for the largest variance in the spectra, with upward peaks likely being emission lines and downward peaks; absorption lines.") 
print("The subsequent components capture progressively smaller contributions to the variance in the dataset.")