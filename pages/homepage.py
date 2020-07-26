"""LiRA Web Home page"""
import streamlit as st

import awesome_streamlit as ast


# pylint: disable=line-too-long
def write():
    """Used to write the page in the app.py file"""
    with st.spinner("Loading Homepage ..."):
        ast.shared.components.title_awesome(" - Homepage")
        #st.title("LiRA Web App - About")
        st.write(
            """Test Home page """
            
        )
        ast.shared.components.video_youtube(
            src="https://www.youtube.com/embed/B2iAodr0fOo"
        )
