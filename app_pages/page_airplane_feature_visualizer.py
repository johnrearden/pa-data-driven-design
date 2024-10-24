import streamlit as st
import pandas as pd
from src.data_management import load_airplane_data, load_pkl_file

# Load the DataFrame from CSV
df_summary_stats = pd.read_csv('outputs/datasets/collection/df_summary_stats.csv')

def page_airplane_feature_visualizer_body():

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
