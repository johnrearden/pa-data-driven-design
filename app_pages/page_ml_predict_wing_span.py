import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from src.data_management import load_airplane_data, load_pkl_file

def page_ml_predict_wing_span_body():  # Renamed function for clarity

    # Load new analysis files
    feature_importance_df = pd.read_pickle("outputs/ml_pipeline/predict_analysis/feature_importance_df.pkl")
    feature_names = pd.read_pickle("outputs/ml_pipeline/predict_analysis/feature_names.pkl")
    
    with open("outputs/ml_pipeline/predict_analysis/mse.txt", 'r') as f:
        mse = f.read()

    with open("outputs/ml_pipeline/predict_analysis/error_analysis.txt", 'r') as f:
        error_analysis = f.read()

    # Load DataFrames first
    try:
        X_test_head = pd.read_csv("outputs/ml_pipeline/predict_analysis/X_test_head.csv")
        X_train_head = pd.read_csv("outputs/ml_pipeline/predict_analysis/X_train_head.csv")
        y_test = pd.read_csv("outputs/ml_pipeline/predict_analysis/y_test.csv")
        y_train = pd.read_csv("outputs/ml_pipeline/predict_analysis/y_train.csv")
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        return  # Exit the function if data files cannot be loaded
    
    st.write("### ML Pipeline: Predict Wing Span")
    # display pipeline training summary conclusions
    st.info(
        f"* The Wing Span predictor computes what Wing Span an airplane need to have in order to"
        f" reach the performance target that you set below. You can select if you want to select"
        f" and incrementally change default values or if you want to input decimal values inside"
        f" (interpolation) of the data range or outside (extrapolation) of the data range."
        f" The relative error (RE) small (less than 0.18%) however be aware that predicting outside of"
        f" the data range (extrapolation) is very unreliable since this is"
        f" effectivly 'unknown territory' with no data to support the assumption of a regression"
    )

    st.write("---")

    st.write("#### Mean Squared Error")
    st.info(mse)

    st.write("#### Mean Error (ME) and Relative Error")
    st.info(error_analysis)

    st.write("#### Feature Importance DataFrame")
    st.dataframe(feature_importance_df)

    st.write("#### Feature Names")
    st.dataframe(feature_names)

    # Loading plots with error handling
    try:
        predicted_vs_actual = plt.imread("outputs/ml_pipeline/predict_analysis/predicted_vs_actual.png")
        st.write("#### Predicted vs Actual Plot")
        st.image(predicted_vs_actual)
    except Exception as e:
        st.error(f"Error loading predicted vs actual plot: {e}")

    try:
        residuals_distribution = plt.imread("outputs/ml_pipeline/predict_analysis/residuals_distribution.png")
        st.write("#### Residuals Distribution Plot")
        st.image(residuals_distribution)
    except Exception as e:
        st.error(f"Error loading residuals distribution plot: {e}")

    try:
        residuals_vs_fitted = plt.imread("outputs/ml_pipeline/predict_analysis/residuals_vs_fitted.png")
        st.write("#### Residuals vs Fitted Plot")
        st.image(residuals_vs_fitted)
    except Exception as e:
        st.error(f"Error loading residuals vs fitted plot: {e}")

    st.write("#### X Test Head")
    st.dataframe(X_test_head)

    st.write("#### X Train Head")
    st.dataframe(X_train_head)

    st.write("#### Y Test")
    st.dataframe(y_test)

    st.write("#### Y Train")
    st.dataframe(y_train)

    st.write("---")

    # If needed, you can still include more details about the model here
    st.write("#### Wingspan Predictor Model Summary")
    wingspan_predictor_model = load_pkl_file("outputs/ml_pipeline/predict_analysis/wingspan_predictor_model.pkl")
    st.write(wingspan_predictor_model)  # Adjust based on what you want to display

# Note: Ensure to call this function in your main app code.
