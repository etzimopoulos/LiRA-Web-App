"""About the Author"""
import streamlit as st

import awesome_streamlit as ast

# pylint: disable=line-too-long
def write():
    """Used to write the page in the main LiRAWeb file"""
    with st.spinner("Loading About ..."):
        ast.shared.components.title_awesome(" - LiRA Web App - About")
        #st.title("Evangelos Tzimopoulos")
        st.markdown(
            """## About Angelo
            Angelo is a Principal Consultant... """
        )
