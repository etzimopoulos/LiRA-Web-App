#!/usr/bin/env python
# coding: utf-8

# # Import Libraries
import streamlit as st

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


st.title('LiRA - The Interactive Linear Regression App')

st.image(image='lirawebapp-image.png',caption='https://pngtree.com/so/graph-icons')
#st.write(<a href='https://pngtree.com/so/graph-icons'>graph-icons png from pngtree.com</a>)

# # A. Solve using Analytical Calculus - Random data points

# ## Create random X and y samples
st.write('To start with we will simulate a Linear Function which will be used to generate our data. In order to do so please select the number of samples:')


st.header('Simulate Linear function - Y = aX + b')
np.random.seed(1)
st.sidebar.subheader('Sampling')
#st.write('Generate Random numbers')
n = st.sidebar.slider('Select number of samples for X and Y', 50, 100)
# Create r and r1, random vectors of 100 numbers each with mean = 0 and standard deviation = 1
r = np.random.randn(n)
r1 = np.random.randn(n)

st.sidebar.subheader('Configure distribution for X')
# Create random Input vector X using r
# Use these for good visual - (mean_x = 3, stddev_x = 2)
mean_x = st.sidebar.slider("Select Mean for generating X",-5,5,3)
stddev_x = st.sidebar.slider('Selct Standard Deviation for generating X',-5,5,2)
X = mean_x * r + stddev_x

st.sidebar.subheader('Coefficients')
#st.write('Select a (Slope) and b (Intercept) for Simulated Linear function')
# Select a = 0.35 and b = 2.5 for good visual
a = st.sidebar.slider('Select "Slope" for Regression line', 0.01, 2.0,0.35)
b = st.sidebar.slider('Select "Intercept" for Regression line', 1.0, 5.0, 2.5)

st.sidebar.subheader('Residual')
#st.write('Select residual distribution for noise added to Simulated Linear function')
# Create random Residual term Res using r
# mean = 0
stddev_res = st.sidebar.slider ('Select Standard Deviation for residual error',0.0,1.0,0.2)
res = stddev_res* r1 


#st.sidebar.[n, mean_x, stddev_x, stddev_res]()
# Generate Y values based on the simulated regression line and error/noise
# Population Regression Line
yreg = b + a * X 
# Adding noise/error
y = yreg + res                  

# Storing Population Regression Line "RegL", data points X and y in a data frame
rl = pd.DataFrame(
    {'X': X,
     'y': y,
     'RegL':yreg}
)

st.write('Show the first five rows of our dataframe')
st.dataframe(rl.head().style.highlight_max(axis=0))



# ## Calculate coefficients alpha and beta

# Assuming y = aX + b
# a ~ alpha
# b ~ beta

# Calculate the mean of X and y
xmean = np.mean(X)
ymean = np.mean(y)

# Calculate the terms needed for the numator and denominator of alpha
rl['CovXY'] = (rl['X'] - xmean) * (rl['y'] - ymean)
rl['VarX'] = (rl['X'] - xmean)**2

# Calculate alpha
# Numerator: Covariance between X and y
# Denominator: Variance of X
alpha = rl['CovXY'].sum() / rl['VarX'].sum()

# Calculate beta
beta = ymean - (alpha * xmean)
print('alpha =', alpha)
print('beta =',beta)


# ## Prediction - Least Squares Line
ypred = alpha * X + beta


# ## Calculate Model Metrics - RSS, RSE(σ), TSS and R^2 Statistic

# Residual Errors
RE = (rl['y'] - ypred)**2
#Residual Sum Squares
RSS = RE.sum()
print("Residual Sum of Squares (RSS) is:",RSS)

# Estimated Standard Variation (sigma) or RSE
RSE = np.sqrt(RSS/(n-2))
print("\nResidual Standar Error (Standard Deviation σ) is:",RSE)

# Total Sum of squares (TSS)
TE = (rl['y'] - ymean)**2
# Total Sum Squares
TSS = TE.sum()
print("\nTotal Sum of Squares (TSS) is:",TSS)

# R^2 Statistic
R2 = 1 - RSS/TSS
print("\n R2 Statistic is:",R2)


# ## Assessing Coefficients accuracy


# Degrees of freedom
df = 2*n - 2

# Standard error, t-Statistic and  p-value for Slope "alpha" coefficient
SE_alpha = np.sqrt(RSE**2/rl['VarX'].sum())
t_alpha = alpha/SE_alpha
p_alpha = 1 - stats.t.cdf(t_alpha,df=df)

# Standard error, t-Statistic and  p-value for Intercept "beta" coefficient
SE_beta = np.sqrt(RSE*(1/n + xmean**2/(rl['VarX'].sum())))
t_beta = beta/SE_beta 
p_beta = 1 - stats.t.cdf(t_beta,df=df)


# ## Coefficients Assessment Summary

st.header('Model Evaluation Metrics')
st.subheader('Assessment of Coefficients')
mds = pd.DataFrame(
    {'Name':['Slope (alpha)', 'Intercept (beta)'],
     'Coefficient': [alpha, beta],
     'RSE':[SE_alpha, SE_beta],
     't-Statistic':[t_alpha, t_beta],
     'p-Value':[p_alpha, p_beta]
    }
)
mds


st.subheader('Model Assessment Summary')

# Model Assessment - Storing all key indicators in dummy data frame with range 1
ms = pd.DataFrame(
    {'Ref': range(0,1),
     'Residual Sum of Squares (RSS)': RSS,
     'RSE (Standard Deviation σ)': RSE,
     'Total Sum of Squares (TSS)': TSS,
     'R2 Statistic': R2
     }
)

# Cut out the dummy index column to see the Results
ms.iloc[:,1:9]    

st.header('Plots')
st.write('Plot Predicted vs Actual vs Sampled Data')

# Plot regression against actual data
plt.figure(figsize=(12, 6))
# Population Regression Line
plt.plot(X,rl['RegL'], label = 'Actual (Population Regression Line)',color='green')
# Least squares line
plt.plot(X, ypred, label = 'Predicted (Least Squares Line)', color='blue')     
# scatter plot showing actual data
plt.plot(X, y, 'ro', label ='Collected data')   
plt.title('Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
st.pyplot()





