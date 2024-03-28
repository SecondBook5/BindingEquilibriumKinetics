import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Read the .csv data file into Python and store the data in a DataFrame
data = pd.read_csv('example1_data.csv')

# Extract x and y data from the DataFrame
x = data.iloc[:, 0].values
y = data.iloc[:, 1].values

# Initial guess for KD (K_D) in nM
k0 = 100  # nM

# Define the fitting function


def binding_equation(x, k):
    """
    Custom equation representing the fraction of bound kinase.

    Parameters:
    x (numpy.ndarray): Concentration of imatinib.
    k (float): Equilibrium dissociation constant (KD).

    Returns:
    numpy.ndarray: Fraction of bound kinase.
    """
    return x / (x + k)


# Fit the data to the custom equation
params, _ = curve_fit(binding_equation, x, y, p0=k0)

# Extract the fitted KD value
kFit = params[0]

# Plot the data and the fit
plt.figure()
plt.plot(x, y, 'bo', label='Data')
plt.plot(x, binding_equation(x, kFit), 'r-', label='Fit')
plt.title('Imatinib Binding Curve')
plt.xlabel('nanomolar [nM]')
plt.ylabel('fraction bound')
plt.legend()

plt.savefig('ImatinibBindingCurve')
plt.show()

# Display the equilibrium dissociation constant (KD)
print(f'Equilibrium Dissociation Constant (KD): {kFit:.4f} nM')
