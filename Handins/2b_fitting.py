import numpy as np
from scipy import linalg # np.linalg
import matplotlib.pyplot as plt
import random as rand
from scipy.optimize import curve_fit

# Function to generate Asin(x) + B + noise
def data(x, A, B):
    np.random.seed(16)
    noise = np.random.normal(0, 0.5, len(x))  # Gaussian noise with mean 0, std 0.5
    return A*np.sin(x) + B + noise

# Initialize data to curve fit with least squares
def initFitting():
    x = np.linspace(0, 100, 1000)
    y = data(x, 4, 2)
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    ones = np.ones(len(x))
    A_sin = np.vstack([sin_x, ones]).T
    A_cos = np.vstack([cos_x, ones]).T
    return x, y, A_sin, A_cos

# Function to find curve fitting parameters of
def sine(x, A, B, phi):
    return A*np.sin(x + phi) + B

# Function to find curve fitting parameters of
def cosine(x, A, B, phi):
    return A*np.cos(x + phi) + B

def main() -> None:

    # Least Square Fitting
    x, y, Asin, Acos = initFitting()
    kSin, rsSin, rankSin, sSin = linalg.lstsq(Asin, y)
    A_kSin, B_kSin = kSin
    print("| curve fitting method | function | parameter | value             |",)
    print("| least squares        | sin(x)   | A         |", A_kSin, "|")
    print("| least squares        | sin(x)   | B         |", B_kSin, "|")

    kCos, rsCos, rankCos, sCos = linalg.lstsq(Acos, y)
    A_kCos, B_kCos = kCos
    print("| least squares        | cos(x)   | A         |", A_kCos, "|")
    print("| least squares        | cos(x)   | B         |", B_kCos, "|")

    # optimize.curve_fitting
    sineOptParams, sineCovParam = curve_fit(sine, x, y)
    A_sineCurve, B_sineCurve, Phi_sineCurve = sineOptParams
    print("| curve_fit            | sin(x)   | A         |", A_sineCurve, "|")
    print("| curve_fit            | sin(x)   | B         |", B_sineCurve, "|")
    print("| curve_fit            | sin(x)   | phi       |", Phi_sineCurve, "|")

    cosOptParams, cosCovParam = curve_fit(cosine, x, y)
    A_cosineCurve, B_cosineCurve, Phi_cosineCurve = cosOptParams
    print("| curve_fit            | cos(x)   | A         |", A_cosineCurve, "|")
    print("| curve_fit            | cos(x)   | B         |", B_cosineCurve, "|")
    print("| curve_fit            | cos(x)   | phi       |", Phi_cosineCurve, "|")

    plt.plot(x, y, label='Noisy Sine Data')
    plt.plot(x, A_kSin*np.sin(x) + B_kSin, '--', label='Lstsq sin(x) fitted')
    plt.plot(x, A_kCos*np.cos(x) + B_kCos, '--', label='Lstsq cos(x) fitted')

    plt.plot(x, sine(x, A_sineCurve, B_sineCurve, Phi_sineCurve), '*', label='Curve fitted sine')
    plt.plot(x, cosine(x, A_cosineCurve, B_cosineCurve, Phi_cosineCurve), '.', label='Curve fitted cosine')

    plt.grid(True)
    plt.legend()
    plt.title('Noisy sine data with least squares fits and curve_fits')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 6)
    plt.show()

if __name__ == '__main__':
    main()
