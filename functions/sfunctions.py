"""About the Author"""
import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


# ****************************************************************************************************
# Intro Logo function for each page
# ****************************************************************************************************
def logo():
   
   st.title('The Interactive Linear Regression App')
   st.image(image='lirawebapp-image1.png',caption='Source: https://pngtree.com/so/graph-icons')
   st.write('''
             
    Welcome to the Interactive Linear Regression App!
   ''') 
   
# ****************************************************************************************************
# Generic function to create random variables based on specific distribution characteristics
# ****************************************************************************************************

def create_distribution(mean, stddev, data):
    var = mean * data + stddev
    return var


# ****************************************************************************************************
# Generic function to create a dependent variable based on Slope, Intercept and Error
# ****************************************************************************************************

def create_variable(a, b, X, err):
    # Actual Regression Line
    y_act = b + a * X 
    # Population line: Actual Regression line fused with noise/error
    y = y_act + err                  
    return y, y_act


# ****************************************************************************************************
# Ordinary Least Squares method to calculate coeffiencts a and b
# ****************************************************************************************************

def OLS_method(rl):
    # ## Calculate coefficients alpha and beta
    
    # Assuming y = aX + b
    # a ~ alpha
    # b ~ beta
    
    # Calculate the mean of X and y
    xmean = np.mean(rl['X'])
    ymean = np.mean(rl['y'])
    
    # Calculate the terms needed for the numerator and denominator of alpha
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
    
    return alpha, beta

# ****************************************************************************************************
# Predict function to generate the Predicted Regression Line
# ****************************************************************************************************

def liraweb_predict(a, b, X, method):
    if method == "OLS":
        y = a * X + b
        return y
    else:
        print("please specify prediction method correctly")



# ****************************************************************************************************************************
# Function to Evaluate the Model Coefficients and Metrics - RSS, RSE(σ), TSS and R^2 Statistic
# ****************************************************************************************************************************

def OLS_evaluation(rl, ypred, alpha, beta, n):
    ymean = np.mean(rl['y'])
    xmean = np.mean(rl['X'])
    
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
    
    # Variance of X
    rl['VarX'] = (rl['X'] - xmean)**2
    
    # Standard error, t-Statistic and  p-value for Slope "alpha" coefficient
    SE_alpha = np.sqrt(RSE**2/rl['VarX'].sum())
    t_alpha = alpha/SE_alpha
    p_alpha = 1 - stats.t.cdf(t_alpha,df=df)
    
    # Standard error, t-Statistic and  p-value for Intercept "beta" coefficient
    SE_beta = np.sqrt(RSE*(1/n + xmean**2/(rl['VarX'].sum())))
    t_beta = beta/SE_beta 
    p_beta = 1 - stats.t.cdf(t_beta,df=df)
    
    
    # Coefficients Assessment Dataframe - Storing all coeffient indicators in dataframe
    mcf_df = pd.DataFrame(
        {'Name':['Slope (alpha)', 'Intercept (beta)'],
         'Coefficient': [alpha, beta],
         'RSE':[SE_alpha, SE_beta],
         't-Statistic':[t_alpha, t_beta],
         'p-Value':[p_alpha, p_beta]
        }
    )
    mcf_df
      
    # Model Assessment Dataframe - Storing all key indicators in dummy dataframe with range 1
    ma_df = pd.DataFrame(
        {'Ref': range(0,1),
         'RSS': RSS,
         'RSE (σ)': RSE,
         'TSS': TSS,
         'R2': R2
         }
    )
    ma_df.iloc[:,1:9]
    return mcf_df, ma_df