

# # Import Libraries
import streamlit as st
#import awesome_streamlit as ast
#from awesome_streamlit import experiments as ste

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

#import pages.homepage
#import pages.about

#ast.core.services.other.set_logging_format()

#PAGES = {
#    "Home": pages.homepage,
#    "About": pages.about,
#}



# ****************************************************************************************************
# Generic function to create random variables based on specific distribution characteristics
# ****************************************************************************************************
@st.cache
def create_distribution(mean, stddev, data):
    var = mean * data + stddev
    return var


# ****************************************************************************************************
# Generic function to create a dependent variable based on Slope, Intercept and Error
# ****************************************************************************************************
@st.cache
def create_variable(a, b, X, err):
    # Actual Regression Line
    y_act = b + a * X 
    # Population line: Actual Regression line fused with noise/error
    y = y_act + err                  
    return y, y_act


# ****************************************************************************************************
# Ordinary Least Squares method to calculate coeffiencts a and b
# ****************************************************************************************************
@st.cache
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

# Predict function to generate the Predicted Regression Line
@st.cache
def liraweb_predict(a, b, X, method):
    if method == "OLS":
        y = a * X + b
        return y
    else:
        print("please specify prediction method correctly")



# ****************************************************************************************************************************
# Function to Evaluate the Model Coefficients and Metrics - RSS, RSE(σ), TSS and R^2 Statistic
# ****************************************************************************************************************************
@st.cache
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



# ***************************************************************************************************************************
# Main function
# ***************************************************************************************************************************    

def main():
    """Main function of the App"""
    #st.sidebar.title('Navigation')
    #selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    
    #page = PAGES[selection]

    #with st.spinner(f"Loading {selection} ..."):
    #    ste.shared.components.write_page(page)

    # **************************************************************************************************************************
    # Introduction Page
    # **************************************************************************************************************************
    st.title('LiRA - The Interactive Linear Regression App')
    st.image(image='lirawebapp-image.png',caption='Source: https://pngtree.com/so/graph-icons')
    st.write('''
             
    Welcome to the Interactive Linear Regression App.
            
    Purpose of this app is to provide a simple interface to experiment with Linear Regression. Using this page you'll be able to:
    
    ####
    ## 1. Simulate a linear function 
    
    Typically, during Linear Regression, we're trying to predict a linear function of type
     Y = aX + b from the data samples. In this case there is no dataset, so we'll create a predefined and configurable
     linear function to generate these data from. In order to do so we need to specify the following:
     
    #### Distribution of X (Input)
    In order to generate random data samples for input X, we'll need to know the Mean and Deviation of the distribution,
    which can be set by adjusting the respective controls in the sidebar widget.
     
    #### Coefficients (a, b)
    The Slope and Intercept of the linear function can also be adjusted using the controls available in the sidebar widget.
    
    ####
    ## 2. Generate Random Data population
    Once we have simulated a linear function, we need to infuse these data with some noise, to allow for the algoright to discover
    the simulated linear function. In order to do so we need to specify number of Samples "n" and the Mean of the error Distribution
    or Mean of Error (Residual):
        
    #### Number of Samples - n
    To generate the data, the number of data needs to be specified as per respective control in the sidebar.
    
    #### Residual - e
    Distribution of error to be added to Y to generate the Y_Samples.      
    ''')
    
    
    # ****************************************************************************************************
    # User Input - Sidebar widgets
    # ****************************************************************************************************
    st.sidebar.title("Intro")
    st.sidebar.info("Welcome to LiRA - the Interactive Linear Regression Application")
    st.sidebar.title("App Controls")
    st.sidebar.subheader('**Number of samples**')
    n = st.sidebar.slider('Select the number of samples for the population',5, 100)
    
    st.sidebar.subheader('Configure distribution for X')
    # Use these for good visual - (mean_x = 3, stddev_x = 2)
    mean_x = st.sidebar.slider("Select Mean for generating X",-10,10,3)
    stddev_x = st.sidebar.slider('Select Standard Deviation for generating X',-5,5,2)
    
    st.sidebar.subheader('Coefficients')
    # Select a = 0.35 and b = 2.5 for good visual
    a = st.sidebar.slider('Select "Slope" for Regression line', -2.0, 2.0,0.15)
    b = st.sidebar.slider('Select "Intercept" for Regression line', -10.0, 10.0, 2.5)
    
    st.sidebar.subheader('Residual')
    #st.write('Select residual distribution for noise added to Simulated Linear function')
    mean_res = st.sidebar.slider ('Select Standard Deviation for residual error',0.0,2.0,0.7)
    
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app is built and maintained by Evangelos Tzimopoulos. 
        
        You can contact me at 
        [LinkedIn](https://www.linkedin.com/in/etzimopoulos/) or view my 
        [GitHub page](https://github.com/etzimopoulos).
        """
    )
    
    # ***************************************************************************************************************************
    # Generate Data
    # ***************************************************************************************************************************
    # Create r and r1, random vectors of "n" numbers each with mean = 0 and standard deviation = 1
    np.random.seed(100)
    r = np.random.randn(n)
    r1 = np.random.randn(n)
    
    # Create Independent Variable as simulated Input vector X by specifying Mean, Standard Deviation and number of samples 
    X = create_distribution(mean_x, stddev_x, r)
    
    # Create Random Error as simulated Residual Error term err using mean = 0 and Standard Deviation from Slider
    err = create_distribution(mean_res,0, r1)
    
    # Create Dependent Variable simulated Output vector y by specifying Slope, Intercept, Input vector X and Simulated Residual Error
    y, y_act = create_variable(a, b, X, err)
    
    st.write('''
    ##
    ## 3. View a sample of generated Data
       The table below, shows a sample of the generated population "X" and "y" along with "Y_act", the actual output of the simulated 
       linear function used to generate the observed "y".
             ''')
    # Storing Population Actual Regression Line "y_act", data points X and y in a data frame
    rl = pd.DataFrame(
        {"X": X,
         'y': y,
         'y_act': y_act}
    )
    st.dataframe(rl.head())
    # ****************************************************************************************************************************


    st.write('''
    ##
    ## 4. Select Linear Regression Method
    In order to implement the linear regression model, there are 4 options available which are summarised below along with links
    for more in depth reading:
    
    * [Ordinary Least Squares - Simple Linear Regression](https://en.wikipedia.org/wiki/Simple_linear_regression)
    * [Ordinary Least Squares - Normal Equations](https://en.wikipedia.org/wiki/Ordinary_least_squares)
    * [Gradient Descent](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/) or [LSM Algorithm](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
    * [SKlearn - Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)

    #### Select Linear Regression method
             ''')
       
    #method1=["Ordinary Least Squares", "Normal Equations", "Gradient Descent", "SKlearn"]
    method=["Ordinary Least Squares"]
    lira_method = st.selectbox('',(method))
    
    
    
    
    
    if lira_method == "Ordinary Least Squares":
        # Calculate coefficients
        alpha, beta = OLS_method(rl)
        # Calculate Regression Line
        ypred = liraweb_predict(alpha, beta, X, "OLS")
    else:
        print("Select OLS for now")

    
    
    
    st.write('''
    ####
    ## 5. Evaluate Model Metrics
    At this section, the predicted model and its coeeficients will be evaluated using various Statical Measures.
        ''')
              
    # Evaluate Model
    model_coeff, model_assess = OLS_evaluation(rl, ypred, alpha, beta, n);
    st.write('''
    #### Assessment of Coefficients
    * Residual Square Error - RSE
    * t-Statistic
    * p-Value
    ''')

    model_coeff
    st.write('''
  
    #### Model Assessment Summary
    * Residual Sum of Squares - RSS
    * RSE (Standard Deviation σ) - RSE
    * Total Sum of Squares - TSS 
    * R2 Statistic
            ''') 
    # Cut out the dummy index column to see the Results
    model_assess.iloc[:,1:9]    
    
    st.write('''
    #### 
     More reading on evaluating the linear regression model can be found [here](https://www.ritchieng.com/machine-learning-evaluate-linear-regression-model/).
            
    ''')
    
    
    st.write('''
    ##
    ## 6. Plot results
             ''')
              
    st.write('''
             Plotting the Predicted Least Squares linear function at the same diagram with the Actual line used to
             generate the data, as well as the Sampled Data, gives a good visual overview of the prediction capability of the model.
             ''')
    
    # Plot regression against actual data
    plt.figure(figsize=(12, 6))
    # Population Regression Line
    plt.plot(X,rl['y_act'], label = 'Actual (Population Regression Line)',color='green')
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



if __name__ == "__main__":
    main()


