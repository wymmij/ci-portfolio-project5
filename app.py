import streamlit as st

from app_pages.multipage import MultiPage


def page_placeholder_body():
    """
    Temporary dashboard landing page while app pages are added.
    """
    st.title("Heritage Housing Issues")

    st.write(
        "The Streamlit dashboard structure is in place. "
        "Dashboard pages will be added incrementally."
    )


app = MultiPage(app_name="Heritage Housing Issues")

app.add_page("Project Summary", page_placeholder_body)

app.run()