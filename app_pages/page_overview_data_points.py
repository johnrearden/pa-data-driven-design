import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_airplane_data

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def page_overview_data_points_body():

    # load data
    df = load_airplane_data()

    # hard copied from churned customer study notebook
    vars_to_study = ['Contract', 'InternetService',  # CHANGE THIS!!!!!!!!!!!
                     'OnlineSecurity', 'TechSupport', 'tenure']  # CHANGE THIS!!!!!!!!!!!

    st.write("### General Aviation data set")
    st.info(
        f"* The client wants to understand the relationship between the Design and Performance parameters"
        f"* for Airplanes that falls into the category of General Aviation so that the client can learn how to"
        f" create quick conceptual aircraft designs drafts for performance specifications.")

    # inspect data
    if st.checkbox("Inspect Customer Base"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))