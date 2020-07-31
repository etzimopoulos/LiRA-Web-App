

# # Import Libraries
import streamlit as st
import awesome_streamlit as ast

import pages.homepage
import pages.about
import pages.resources

ast.core.services.other.set_logging_format()

PAGES = {
    "Home": pages.homepage,
    "Resources": pages.resources,
    "About": pages.about,
}



def main():
    """Main function of the App"""
    st.title('The Interactive Linear Regression App')
    #st.image(image='lirawebapp-image.png',caption='Source: https://pngtree.com/so/graph-icons')
    
    
    st.sidebar.title("Intro")
    st.sidebar.info("Welcome to LiRA - the Interactive Linear Regression Application")
    
    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))
    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)

    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app is built and maintained by Evangelos Tzimopoulos. 
        
        You can contact me at 
        [LinkedIn](https://www.linkedin.com/in/etzimopoulos/) or view my 
        [GitHub page](https://github.com/etzimopoulos).
"""
    )

if __name__ == "__main__":
    main()


