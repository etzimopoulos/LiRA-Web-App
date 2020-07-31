"""Home page shown when the user enters the application"""
import streamlit as st

#import awesome_streamlit as ast


# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    st.image(image='angelo_pixel1.png',caption='by Angelo')
    with st.spinner("Loading About ..."):
        st.markdown(
            """
        This app is built and maintained by Evangelos Tzimopoulos. 
        
        You can contact me at 
        [LinkedIn](https://www.linkedin.com/in/etzimopoulos/) or view my 
        [GitHub page](https://github.com/etzimopoulos).
            """
        )