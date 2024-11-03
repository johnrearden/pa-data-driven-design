import streamlit as st
import pandas as pd

from src.data_management import load_airplane_data, load_pkl_file

df_summary_stats = pd.read_csv('outputs/datasets/collection/df_summary_stats.csv')  # Load the DataFrame from CSV

def page_get_to_know_the_dataset_body():
    """
    Displays an interactive page in the Streamlit app to explore a dataset related to airplanes.

    This function loads the airplane dataset and summary statistics from CSV files, allowing users
    to select a feature from a dropdown menu. For the selected feature, it displays images representing
    the minimum, mean, and maximum values along with their corresponding models and companies.
    
    The Images where created based on analysis in notebook 2_A_domain_specific_analysis.
    
    Users can also choose to inspect the full airplane dataset, which shows the number of rows and
    columns, as well as the first 10 rows of data.

    Features:
        - Dropdown menu for selecting a feature based on summary statistics.
        - Visual representation of the selected feature's minimum, mean, and maximum values.
        - Option to inspect the underlying airplane dataset.

    Note:
        This function requires the following libraries:
        - Streamlit
        - Pandas
        - Custom functions from `src.data_management` for loading data.

    Outputs:
        - Displays images and information based on user selections.
        - Displays a preview of the dataset when requested.
    """
    st.write("### Get to know the dataset")

    # load data
    df = load_airplane_data()
    
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
        st.image(min_image_path, caption=f"{row['MIN_model']} ({row['MIN_company']})\nMIN: {row['MIN_value']} {row['units']}")
    
    with col2:
        st.image(mean_image_path, caption=f"{row['MEAN_model']} ({row['MEAN_company']})\nMEAN: {row['MEAN_value']} {row['units']}")

    with col3:
        st.image(max_image_path, caption=f"{row['MAX_model']} ({row['MAX_company']})\nMAX: {row['MAX_value']} {row['units']}")

    # inspect data
    if st.checkbox("Inspect Airplane data"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))
