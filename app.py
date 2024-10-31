import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_airplane_feature_explanation import page_airplane_feature_explanation_body
from app_pages.page_get_to_know_the_dataset import page_get_to_know_the_dataset_body
from app_pages.page_multi_engine_airplane_study import page_multi_engine_airplane_study_body
from app_pages.page_wing_span_predictor import page_wing_span_predictor_body
from app_pages.page_regression_playground import page_regression_playground_body
from app_pages.page_domain_specific_analysis import page_domain_specific_analysis_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
from app_pages.page_ml_predict_engine_type import page_ml_predict_engine_type_body
from app_pages.page_ml_predict_wing_span import page_ml_predict_wing_span_body
from app_pages.page_ml_cluster import page_ml_cluster_body

app = MultiPage(app_name= "Airplane Performance Predictor") # Create an instance of the app 

# Add pages to the MultiPage app
app.add_page("Quick Project Summary", page_summary_body)
app.add_page("Airplane Feature Explanation", page_airplane_feature_explanation_body)
app.add_page("Get to know the dataset", page_get_to_know_the_dataset_body)
app.add_page("Multi Engine Airplane Study", page_multi_engine_airplane_study_body)
app.add_page("Wing Span Predictor", page_wing_span_predictor_body)
app.add_page("Regression Playground", page_regression_playground_body)
app.add_page("Domain Specific Analysis", page_domain_specific_analysis_body)
app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
app.add_page("ML: Predict Engine Type", page_ml_predict_engine_type_body)
app.add_page("ML: Predict Wing Span", page_ml_predict_wing_span_body)
app.add_page("ML: Cluster Analysis", page_ml_cluster_body)

app.run() # Run the app
