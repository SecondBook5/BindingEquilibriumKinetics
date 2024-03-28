import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Read the .csv data file into Python and store the data in a DataFrame
data = pd.read_csv('example2_data.csv')

# Extract x and y data from the DataFrame
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

# Plot the data
plt.figure()
plt.plot(x, y, marker='o', linestyle='', label='Data')
plt.xlabel('[ATP](nM)', fontsize=12)
plt.ylabel('Fluorescence (au)', fontsize=12)
plt.title('ATP Binding Curve')
plt.legend()

# Mark the potential saturation point
saturating_concentration = 20000
fluorescence_at_saturation = y[np.where(x == saturating_concentration)]
plt.plot(saturating_concentration, fluorescence_at_saturation,
         'ro', markersize=10, label='Potential Saturation Point')
plt.legend()

# Identify baseline fluorescence (b0)
b0 = y[np.where(x == 0)]

# Identify signal adjustment (a0)
a0 = fluorescence_at_saturation - b0

# Identify initial estimate for KD (k0)
half_max_fluorescence = b0 + (0.5 * a0)
index_closest = np.argmin(np.abs(y - half_max_fluorescence))
k0 = x[index_closest]

# Define the fitting function


def binding_equation(x, b, a, k):
    """
    Custom equation representing the fraction of bound kinase.

    Parameters:
    x (numpy.ndarray): Concentration of ATP.
    b (float): Baseline fluorescence.
    a (float): Signal adjustment.
    k (float): Equilibrium dissociation constant (KD).

    Returns:
    numpy.ndarray: Fraction of bound kinase.
    """
    return b + a * (x / (x + k))


# Fit the data to the custom equation
params, _ = curve_fit(binding_equation, x, y, p0=[b0[0], a0[0], k0])

# Extract the fitted KD value
kFit = params[2]


# Plot the fit
plt.figure()
plt.plot(x, y, marker='o', linestyle='', label='Data')
plt.plot(x, binding_equation(x, *params), 'r-', label='Fit')
plt.xlabel('[ATP](nM)', fontsize=12)
plt.ylabel('Fluorescence (au)', fontsize=12)
plt.title('ATP Binding Curve')
plt.legend()

# Set x-axis tick labels
plt.gca().set_xticks([0, 5e4, 10e4, 15e4])
plt.gca().set_xticklabels(['0', '50,000', '100,000', '150,000'])

# Display the equilibrium dissociation constant (KD)
print(f'Equilibrium Dissociation Constant (KD): {kFit:.4f}')

plt.show()
