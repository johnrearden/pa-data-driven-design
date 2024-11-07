import streamlit as st

# Class to generate multiple Streamlit pages using an object oriented approach


class MultiPage:
    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name)


    def add_page(self, title, func) -> None:
        self.pages.append({"title": title, "function": func})

    def run(self):
        # Display the DDD logo
        st.sidebar.markdown("&nbsp;" * 2)
        st.sidebar.image("images_dashboard/data_driven_design_logo_300.png", width=250)
        st.sidebar.write("")
        st.sidebar.write("")
        st.sidebar.write("")

        st.title(self.app_name)
        page = st.sidebar.radio('Menu', self.pages, format_func=lambda page: page['title'])
        page['function']()
