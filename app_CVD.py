import streamlit as st
from app_pages.multipage import MultiPage

from app_pages.page_project_summary import page_project_summary_body
from app_pages.page_project_hypotheses import page_project_hypotheses_body
from app_pages.page_feat_correlation import page_feat_correlation_body
from app_pages.page_heartdisease_prediction import page_heartdisease_pred_body
from app_pages.page_model_performance import page_model_performance_body
from app_pages.page_project_conclusions import page_project_conclusions_body


app = MultiPage(app_name="CVD Predictor")

app.app_page("Project Summary", page_project_summary_body)
app.app_page("Project Hypotheses", page_project_hypotheses_body)
app.app_page("Feature Correlation Study", page_feat_correlation_body)
app.app_page("Heart Disease Prediction", page_heartdisease_pred_body)
app.app_page("Model Performance", page_model_performance_body)
app.app_page("Project Conclusions", page_project_conclusions_body)

app.run()