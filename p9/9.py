import pandas as pd
import numpy as np1
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures

# Read the dataset
data = pd.read_csv('dataset.csv')
bill = np1.array(data.total_bill)
tip = np1.array(data.tip)

# Initialize Ridge with a polynomial of degree 2
poly = PolynomialFeatures(degree=2)
model = Ridge(alpha=1.0)

# Transform the x values to polynomial values
bill_poly = poly.fit_transform(bill.reshape(-1, 1))

# Fit the model with the polynomial values
model.fit(bill_poly, tip)

# Predict the values for the original x values
ypred = model.predict(bill_poly)

# Plot the results
plt.scatter(bill, tip, color='blue')
plt.plot(bill, ypred, color='red', linewidth=1)
plt.xlabel('Total bill')
plt.ylabel('tip')
plt.show()