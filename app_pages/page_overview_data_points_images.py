import streamlit as st
import pandas as pd
from src.data_management import load_airplane_data, load_pkl_file

# Load the DataFrame from CSV
df_summary_stats = pd.read_csv('outputs/datasets/collection/df_summary_stats.csv')

def page_overview_data_points_images_body():
    st.write("### Prospect Churnometer Interface")
    st.info(
        f"* The client is interested in determining whether or not a given prospect will churn. "
        f"If so, the client is interested to know when. In addition, the client is "
        f"interested in learning from which cluster this prospect will belong in the customer base. "
        f"Based on that, present potential factors that could maintain and/or bring  "
        f"the prospect to a non-churnable cluster."
    )
    st.write("---")
    
    # display_airplane_features() Probably remove this, cant remember from where it comes from


# Streamlit app
st.title("Airplane Feature Visualizer")

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
