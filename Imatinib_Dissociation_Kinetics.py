import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Read the .csv data file into Python and store the data in a DataFrame
data = pd.read_csv('example3_data.csv')

# Extract x and y data from the DataFrame
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

# Plot the data
plt.figure()
plt.plot(x, y, marker='o', linestyle='', label='Data')
plt.xlabel('time (s)', fontsize=12)
plt.ylabel('Fluorescence (au)', fontsize=12)
plt.title('BCR-ABL1/Imatinib Dissociation Curve')
plt.legend()

# Identify baseline fluorescence (b0)
b0 = y[-1]

# Identify signal adjustment (a0)
a0 = y[0] - b0

# Identify initial estimate for k0
half_max_fluorescence = max(y) / 2
half_life_time = interp1d(y, x)(half_max_fluorescence)
k0 = -np.log((half_max_fluorescence - b0) / a0) / half_life_time

# Define the fitting function


def dissociation_equation(x, a, k, b):
    """
    Custom equation representing the dissociation kinetics.

    Parameters:
    x (numpy.ndarray): Time points.
    a (float): Amplitude.
    k (float): Rate constant.
    b (float): Baseline.

    Returns:
    numpy.ndarray: Predicted fluorescence values.
    """
    return a * np.exp(-k * x) + b


# Fit the data to the custom equation
params, _ = curve_fit(dissociation_equation, x, y, p0=[a0, k0, b0])

# Plot the fit
plt.figure()
plt.plot(x, y, marker='o', linestyle='', label='Data')
plt.plot(x, dissociation_equation(x, *params), 'r-', label='Fit')
plt.xlabel('time (seconds)')
plt.ylabel('fluorescence (au)')
plt.legend()

plt.show()
