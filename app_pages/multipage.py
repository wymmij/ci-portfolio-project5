import streamlit as st


class MultiPage:
    """
    Class to generate a multipage Streamlit app.
    """

    def __init__(self, app_name):
        self.pages = []
        self.app_name = app_name

    def add_page(self, title, function):
        """
        Add page to the app.
        """
        self.pages.append({
            "title": title,
            "function": function
        })

    def run(self):
        """
        Run selected app page.
        """
        st.set_page_config(
            page_title=self.app_name,
            page_icon="🏠",
            layout="wide"
        )

        st.sidebar.title(self.app_name)

        page = st.sidebar.radio(
            "Navigation",
            self.pages,
            format_func=lambda page: page["title"]
        )

        st.sidebar.info(
            "This dashboard explores house sale prices and predicts sale "
            "prices for inherited houses."
        )

        page["function"]()