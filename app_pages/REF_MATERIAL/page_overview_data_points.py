import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_airplane_data
df_summary_stats = pd.read_csv('outputs/datasets/collection/df_summary_stats.csv')  # Load the DataFrame from CSV
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def page_overview_data_points_body():

    # load data
    df = load_airplane_data()

    st.write("### General Aviation data set")
    st.info(
        f"* The client wants to understand the relationship between the Design and Performance parameters"
        f"* for Airplanes that falls into the category of General Aviation so that the client can learn how to"
        f" create quick conceptual aircraft designs drafts for performance specifications.")

    st.write("### Airplane Feature Visualizer")

    # Dropdown menu for selecting a feature
    feature = st.selectbox("Select a Feature", df_summary_stats['FEATURE'].unique())

    # Fetch the row corresponding to the selected feature
    row = df_summary_stats[df_summary_stats['FEATURE'] == feature].iloc[0]

    # Construct image paths
    min_image_path = f'images_dashboard/min/min_{feature}.jpg'
    mean_image_path = f'images_dashboard/mean/mean_{feature}.jpg'
    max_image_path = f'images_dashboard/max/max_{feature}.jpg'

    # Display images and corresponding information
    col1, col2, col3 = st.columns(3)

    with col1:
        st.image(min_image_path, caption=f"{row['MIN_model']} ({row['MIN_company']})\nMIN: {row['MIN_value']}")
    
    with col2:
        st.image(mean_image_path, caption=f"{row['MEAN_model']} ({row['MEAN_company']})\nMEAN: {row['MEAN_value']}")
    
    with col3:
        st.image(max_image_path, caption=f"{row['MAX_model']} ({row['MAX_company']})\nMAX: {row['MAX_value']}")

    # inspect data
    if st.checkbox("Inspect Customer Base"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))