
import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
#import time
import altair as alt


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



# ***************************************************************************************************************************
# Generate Data
# ***************************************************************************************************************************
def generate_data(n, a, b, mean_x, stddev_x, mean_res):
       # Create r and r1, random vectors of "n" numbers each with mean = 0 and standard deviation = 1
        np.random.seed(100)
        r = np.random.randn(n)
        r1 = np.random.randn(n)

        # Create Independent Variable as simulated Input vector X by specifying Mean, Standard Deviation and number of samples 
        X = create_distribution(mean_x, stddev_x,r)
        

        # Create Random Error as simulated Residual Error term err using mean = 0 and Standard Deviation from Slider
        err = create_distribution(mean_res,0, r1)
                
        # Create Dependent Variable simulated Output vector y by specifying Slope, Intercept, Input vector X and Simulated Residual Error
        y, y_act = create_variable(a, b, X, err)
        
        # ************ Normal Equations data trasnformations ****
        # Transform "X" to use as matrix in Normal Equations
        X1 = np.matrix([np.ones(n), X]).T
        # Transform "y" to use as matrix in Normal Equations
        y1 = np.matrix(y).T
        
        # ************ Gradient descent
        
        
        
        # Storing Population Actual Regression Line "y_act", data points X and y in a data frame
        rl = pd.DataFrame(
            {"X": X,
             'y': y,
            'y_act': y_act}
        )
        return rl, X1, y1


# ****************************************************************************************************
# Normal Equations method to calculate coeffiencts a and b
# ****************************************************************************************************

def NE_method(X1, y1):
    # Calculate coefficients alpha (or a) and beta (or b)
    A = np.linalg.inv(X1.T.dot(X1)).dot(X1.T).dot(y1)
    a = A[1].item()
    b = A[0].item()
    return a, b



# *****************************************************************************************
# Gradiend Descent Error function    
# *****************************************************************************************

def GD_error(a, b, X, y):
    y_pred = a * X + b  # The current predicted value of Y
    cost = sum((y_pred-y)**2) / (2*len(y)) 
    return cost
         

# *****************************************************************************************
# Gradiend Descent Weights Calculation function    
# *****************************************************************************************
   
def GD_theta(a, b, X, y, L, n):  
    y_pred = a * X + b  # The current predicted value of Y
    # Theta or Weights calculation
    D_a = (-2/n) * sum(X * (y - y_pred))  # Derivative wrt a
    D_b = (-2/n) * sum(y - y_pred)  # Derivative wrt b
    a = a - L * D_a  # Update alpha
    b = b - L * D_b  # Update beta
    return a, b, y_pred


# *****************************************************************************************
# Animation function to show how Gradient Descent predicted line converges to Actual Line    
# *****************************************************************************************

def GD_animate(rl,draw_fig1, data_points, regression_line):
    
   #if mode == "Altair":
    
       
    predicted_line = alt.Chart(rl).mark_line(color='purple').encode(x='X', y='y_pred')
    #draw_fig1.altair_chart(predicted_line)
    draw_fig1.altair_chart(data_points+regression_line+predicted_line)
    
    #err_df = pd.DataFrame(tmp_err)
    #error_line = alt.Chart(err_df).mark_line(color='red').encode(x='epochs', y='err')
    #draw_fig2 = st.altair_chart(error_line)
    #draw_fig2.altair_chart(error_line)
    
    # elif mode == "Matplotlib":
    #     ax.plot(X, y_pred, label = 'Predicted (Least Squares Line)', color='purple')
    #     the_plot.pyplot(plt,clear_figure=False)
        

# ****************************************************************************************************
# Gradient Descent method to calculate coeffiencts a and b
# ****************************************************************************************************
def GD_method(rl, L, epochs):
    
    X=rl['X'] 
    y=rl['y']
    n = float(len(X)) # Number of elements in X
    y_pred = [0]*len(X) # Used in animation with Matplotlib
    rl['y_pred'] = [0]*len(X) # Used in animation with Altair
    
    
    # Plot using Altair for Animation 
    # if mode == "Matplotlib":
    # Define lines
    data_points = alt.Chart(rl).mark_point(color='red').encode(x='X', y='y')
    regression_line = alt.Chart(rl).mark_line(color='green').encode(x='X', y='y_act')
    #error_line = alt.Chart(error).mark_line(color='blue').encode(x='epochs', y='err')
             
    # Draw plots
    draw_fig1 = st.altair_chart(data_points+regression_line)
    #draw_fig2 = st.altair_chart(error_line)
    
    
    # elif mode == "Matplotlib":
    #     fig, ax = plt.subplots()
    #     ax.plot(X,rl['y_act'], label = 'Actual (Population Regression Line)',color='green')
    #     ax.plot(X, y, 'ro', label ='Collected data')   
    #     ax.plot(X, y_pred, label = 'Predicted (Least Squares Line)', color='purple')
    #     ax.set_title('Actual vs Predicted')
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('y')
    #     ax.legend()
    #     the_plot = st.pyplot(plt,clear_figure=False)
    
        
    
    # ************************************************************
    # Performing Gradient Descent 
    # ************************************************************
    # Initialise a and b and error
    a=0
    b=0
    tmp_err = []
    tmp_theta = []
    
    
    # Initialise progress bar
    my_bar = st.progress(0)       
    status_text = st.empty()
    pb_i = round(epochs/100)
    
    for i in range(epochs): 
        a, b, y_pred = GD_theta(a, b, X, y, L, n)
        err = GD_error(a, b, X, y)
       
        # Store cumulative error and weights
        tmp_err.append([i, err]) 
        tmp_theta.append([i, a, b])
        
        
        rl['y_pred'] = a * X + b # To use for animation with Altair
        #error['err'] = err # To use for animation with Altair
        #rl['epochs'] = epochs
        
        # ***********************************************
        # ANIMATE Gradient Descent
        # ***********************************************
        # Animate sampled plots as algorithm converges along with progress bar
        if((i % pb_i) == 0 and round(i/pb_i)<101):
            GD_animate(rl,draw_fig1, data_points, regression_line)
            my_bar.progress(round(i/pb_i))
    status_text.text('Gradient Descent converged to the optimal values. Exiting...')    
    return a, b, y_pred, tmp_err

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
    return alpha, beta

# ****************************************************************************************************
# Predict function to generate the Predicted Regression Line
# ****************************************************************************************************

def liraweb_predict(a, b, X, method):
    if method == "OLS-Simple Linear Regression" or method == "OLS-Normal Equations" or method == "SKlearn":
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
    
    
    print('\n****************************************************')
    print('Estimated Model')
    print('****************************************************')
    print('\nAlpha (Slope) calculated as ', alpha)
    print('\nBeta (Intercept) calculated as ',beta)
    print('\nLeast Squares line predicted as y = %.2fX + %.2f' % (alpha, beta))
    
    
    print('\n****************************************************')
    print('Model performance metrics')
    print('****************************************************')
    # Residual Errors
    RE = (rl['y'] - ypred)**2
    
    #Residual Sum Squares
    RSS = RE.sum()
    print("\nResidual Sum of Squares (RSS) is:",RSS)
    
    # Estimated Standard Variation (sigma) or RSE
    RSE = np.sqrt(RSS/(n-2))
    print("\nResidual Standard Error (Standard Deviation σ) is:",RSE)
    
    # Total Sum of squares (TSS)
    TE = (rl['y'] - ymean)**2
    # Total Sum Squares
    TSS = TE.sum()
    print("\nTotal Sum of Squares (TSS) is:",TSS)
    
    # R^2 Statistic
    R2 = 1 - RSS/TSS
    print("\nR2 Statistic is:",R2)
    
    
    # Assessing Coefficients accuracy
       
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

def plot_GD_error(error):
    plt.figure(figsize=(12, 6))
    # Population Regression Line
    plt.plot(error[1], label='Gradient Descent Method - Error tracking', color='orange')
    plt.title('Error')
    plt.xlabel('Epochs')
    plt.ylabel('error')
    plt.legend()
    st.pyplot()

def plot_model(rl,ypred, method):
    # Plot regression against actual data
    plt.figure(figsize=(12, 6))
    # Population Regression Line
    plt.plot(rl['X'],rl['y_act'], label = 'Actual (Population Regression Line)',color='green')
    # Least squares line
    if method == "OLS-Simple Linear Regression":
        plt.plot(rl['X'], ypred, label = 'Predicted (Least Squares Line)', color='blue')     
    elif method == "OLS-Normal Equations":
        plt.plot(rl['X'], ypred, label = 'Predicted (Least Squares Line)', color='orange')             
    elif method == "Gradient Descent":
        plt.plot(rl['X'], ypred, label = 'Predicted (Least Squares Line)', color='purple')             
   
    elif method == "SKlearn":
        plt.plot(rl['X'], ypred, label = 'Predicted (Least Squares Line)', color='k')
    # scatter plot showing actual data
    plt.plot(rl['X'], rl['y'], 'ro', label ='Collected data')   
    plt.title('Actual vs Predicted')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    #plt.show()
    st.pyplot()
    
def plots_and_metrics(rl, ypred, lira_method,model_coeff, model_assess):
    st.write('''
    ##
    ## 5. Plot results
             ''')
    st.write('''
             The plot gives a good visual overview of the prediction capability of the model, capturing the following elements:
              1) The Predicted Least Squares linear function
              2) The Actual line used to generate the data
              3) The Sampled Data
    ''')

    # Plot results
    plot_model(rl,ypred, lira_method)
    
    st.write('''
    ####
    ## 6. Evaluate Model Metrics
    At this section, the predicted model and its coeeficients will be evaluated using various Statical Measures.
        ''')
    st.write('''
    #### Assessment of Coefficients
    * Residual Square Error - RSE
    * t-Statistic
    * p-Value
    ''')
    st.write(model_coeff)
    st.write('''
    ####
    #### Model Assessment Summary
    * Residual Sum of Squares - RSS
    * RSE (Standard Deviation σ) - RSE
    * Total Sum of Squares - TSS 
    * R2 Statistic
            ''') 
    # Cut out the dummy index column to see the Results
    st.write(model_assess.iloc[:,1:9])
    st.write('''
    #### 
     More reading on evaluating the linear regression model can be found [here](https://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/).
            
    ''')
    
def GD_plots_and_metrics(rl, ypred, error, lira_method, model_coeff, model_assess):
    st.write('''
    ##
    ## 5. Plot results
              ''')
    st.write('''
              The plot gives a good visual overview of the prediction capability of the model, capturing the following elements:
              1) The Predicted Least Squares linear function
              2) The Actual line used to generate the data
              3) The Sampled Data
              
              ''')

    # Plot results
    plot_model(rl, ypred, lira_method)
    
    st.write('''
              The graph below shows the error curve as the model learns to approximate the Predicted line better.
              ''')

    
    plot_GD_error(error)
    
    st.write('''
    ####
    ## 6. Evaluate Model Metrics
    At this section, the predicted model and its coeeficients will be evaluated using various Statical Measures.
        ''')
    st.write('''
    #### Assessment of Coefficients
    * Residual Square Error - RSE
    * t-Statistic
    * p-Value
    ''')
    st.write(model_coeff)
    st.write('''
    ####
    #### Model Assessment Summary
    * Residual Sum of Squares - RSS
    * RSE (Standard Deviation σ) - RSE
    * Total Sum of Squares - TSS 
    * R2 Statistic
            ''') 
    # Cut out the dummy index column to see the Results
    st.write(model_assess.iloc[:,1:9])
    st.write('''
    #### 
      More reading on evaluating the linear regression model can be found [here](https://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/).
            
    ''')