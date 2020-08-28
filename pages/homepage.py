"""LiRA Web Home page"""
import streamlit as st
import pandas as pd
import functions.sfunctions as sf


# pylint: disable=line-too-long
def write():
    #"""Used to write the page in the app.py file"""
    with st.spinner("Loading Homepage ..."):
        #ast.shared.components.title_awesome(" - Homepage")
        #st.title("LiRA Web App - HomePage")
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
        To generate the data, the number of data points need to be specified as per respective control in the sidebar.
        
        #### Residual - e
        Distribution of error to be added to Y to generate the Y_Samples.      
        ''')
        
        # ****************************************************************************************************
        # Input widgets
        # ****************************************************************************************************
        
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
        
        st.sidebar.subheader('Residual Error')
        #st.write('Select residual distribution for noise added to Simulated Linear function')
        mean_res = st.sidebar.slider ('Select Mean for Residual Error',0.0,2.0,0.7)
                        
        # Dataframe to store generated data
        rl, X1, y1 = sf.generate_data(n, a, b, mean_x, stddev_x, mean_res)
        
        st.write('''
        ##
        ## 3. View a sample of generated Data
           The table below, shows a sample of the generated population "X" and "y" along with "Y_act", the actual output of the simulated 
           linear function used to generate the observed "y".
                 ''')
    
        st.dataframe(rl.head())
        # ****************************************************************************************************************************
    
    
        st.write('''
        ##
        ## 4. Select Linear Regression Method
        In order to implement the linear regression model, there are 4 options available:
        
        * Ordinary Least Squares - Simple Linear Regression
        * Ordinary Least Squares - Normal Equations
        * Gradient Descent or LSM Algorithm
        * SKlearn - Linear Models
        
        For more in depth reading about these methods, please check the Resources pages.
        #### Select Linear Regression method
                 ''')
           
        method=["Gradient Descent", "OLS-Simple Linear Regression", "OLS-Normal Equations", "SKlearn"]
        lira_method = st.selectbox('',(method))
        #if st.button('Predict'):
        
        if lira_method == "Gradient Descent":
            # Configuration Parameters for Gradient Descent
            L = st.slider('Select the Learning Rate', 0.0,0.05, 0.015,0.0001)
            epochs = st.slider('Select the number of iterations (Epochs)', 100, 1000,250,5)
            #pmethod = ['Altair','Plotly','Matplotlib', 'Skip Animation']
            #mode = st.selectbox("Select Plotting Library",(pmethod))
            
            # Calculate model and coefficients
            alpha, beta, ypred, tmp_err= sf.GD_method(rl, L, epochs)
            error = pd.DataFrame(tmp_err)
                        
            # Evaluate Model
            model_coeff, model_assess = sf.OLS_evaluation(rl, ypred, alpha, beta, n);
            
            # Results summary
            # Plot final graphs and Evaluate Model metrics
            sf.GD_plots_and_metrics(rl, ypred, error, lira_method,model_coeff, model_assess)
            
            
        
        if lira_method == "OLS-Simple Linear Regression":
            # Calculate coefficients
            alpha, beta = sf.OLS_method(rl)
            
            # Calculate Regression Line
            ypred = sf.liraweb_predict(alpha, beta, rl['X'], lira_method)
            
            # Evaluate Model
            model_coeff, model_assess = sf.OLS_evaluation(rl, ypred, alpha, beta, n);
           
            # Results summary
            # Plot final graphs and Evaluate Model metrics
            sf.plots_and_metrics(rl, ypred, lira_method,model_coeff, model_assess)
            
            
        if lira_method == "OLS-Normal Equations":
            # Calculate coefficients
            alpha, beta = sf.NE_method(X1,y1)
            
            # Calculate Regression Line
            ypred = sf.liraweb_predict(alpha, beta, rl['X'], lira_method)
            
            # Evaluate Model
            # create new evaluation method sf.NE_evaluation and replace - for now use ypred
            model_coeff, model_assess = sf.OLS_evaluation(rl, ypred, alpha, beta, n);
            
            # Results summary
            # Plot final graphs and Evaluate Model metrics
            sf.plots_and_metrics(rl, ypred, lira_method,model_coeff, model_assess)
            
        
        if lira_method == 'SKlearn':
            # Import library
            from sklearn.linear_model import LinearRegression
            
            # Calculate model and coefficients
            lr = LinearRegression()
            lr.fit(rl['X'].values.reshape(-1, 1), rl['y'].values.reshape(-1, 1))
            alpha = lr.coef_[0][0]
            beta = lr.intercept_[0]
            ypred = sf.liraweb_predict(alpha, beta, rl['X'], lira_method)
            
            # Evaluate Model
            model_coeff, model_assess = sf.OLS_evaluation(rl, ypred, alpha, beta, n);
            
            # Results summary
            # Plot final graphs and Evaluate Model metrics
            sf.plots_and_metrics(rl, ypred, lira_method,model_coeff, model_assess)
            

if __name__ == "__main__":
    write()