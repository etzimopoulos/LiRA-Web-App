"""About the Author"""
import streamlit as st


# pylint: disable=line-too-long
def write():
   st.markdown(
            '''
        ## Resources
        Links to more reading about the 4 methods implemented for Linear Regression
        * [Ordinary Least Squares - Simple Linear Regression](https://en.wikipedia.org/wiki/Simple_linear_regression)
        * [Ordinary Least Squares - Normal Equations](https://en.wikipedia.org/wiki/Ordinary_least_squares)
        * [Gradient Descent](https://www.geeksforgeeks.org/gradient-descent-in-linear-regression/) or [LSM Algorithm](http://cs229.stanford.edu/notes/cs229-notes1.pdf)
        * [SKlearn - Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)
    
            
            '''
    )
if __name__ == "__main__":
    write()