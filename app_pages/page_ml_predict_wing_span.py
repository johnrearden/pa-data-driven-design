import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from src.data_management import load_airplane_data, load_pkl_file


def page_ml_predict_wing_span_body():

    # Load new analysis files
    feature_importance_df = pd.read_pickle("outputs/ml_pipeline/predict_analysis/feature_importance_df.pkl")
    feature_names = pd.read_pickle("outputs/ml_pipeline/predict_analysis/feature_names.pkl")

    # Load Files
    try:
        X_test_head = pd.read_csv("outputs/ml_pipeline/predict_analysis/X_test_head.csv")
        X_train_head = pd.read_csv("outputs/ml_pipeline/predict_analysis/X_train_head.csv")
        y_test = pd.read_csv("outputs/ml_pipeline/predict_analysis/y_test.csv")
        y_train = pd.read_csv("outputs/ml_pipeline/predict_analysis/y_train.csv")
        error_analysis = pd.read_csv("outputs/ml_pipeline/predict_analysis/error_analysis.txt")
    except Exception as e:
        st.error(f"Error loading data files: {e}")
        return  # Exit the function if data files cannot be loaded

    st.write("### ML Pipeline: Predict Wing Span")
    # display pipeline training summary conclusions
    st.info(
        f"We set a target metric for our Regressor model: a Relative Error "
        f"(RE = RMSE / mean of y_test) of less than 10%. The achieved "
        f"Relative Error of 5.97% (off by 3.6 feet/1 m for a 60 ft/19 m wing span) "
        f"cleared the target with a margin. "
        f"This target was set deliberately low, "
        f"considering both the diversity of the dataset and the fact that "
        f"the model was designed for a conceptual rather than a detailed "
        f"design user.   \n"
        f"We use Relative Error instead of the more common "
        f"R-Squared because this scaled metric provides a more intuitive "
        f"and practical way to assess the accuracy of the model. R-Squared, "
        f"indicates a moderate performance with a score of 0.5. This "
        f"relatively low score could be partially attributed to the large "
        f"number of features in the model. However, we also need to take "
        f"the context into account when interpreting the R-Squared score."
        )

    st.write("#### Error Analysis")
    st.dataframe(error_analysis)

    st.write("---")

    st.write("#### Wingspan Predictor Model Summary")
    wingspan_predictor_model = load_pkl_file("outputs/ml_pipeline/predict_analysis/wingspan_predictor_model.pkl")
    st.write(wingspan_predictor_model)

    st.write("---")

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

    try:
        confusion_matrix = plt.imread("outputs/ml_pipeline/predict_analysis/confusion_matrix.png")
        st.write("#### Confusion Matrix")
        st.write("##### Only the '16 - 33.6' and '33.6 - 51.2' buckets are showing")
        st.image(confusion_matrix)
    except Exception as e:
        st.error(f"Error loading confusion_matrix: {e}")

    st.write("---")

    st.write("#### Feature Importance DataFrame")
    st.dataframe(feature_importance_df)

    st.write("#### X Test")
    st.dataframe(X_test_head)

    st.write("#### X Train")
    st.dataframe(X_train_head)

    st.write("#### Y Test")
    st.dataframe(y_test)

    st.write("#### Y Train")
    st.dataframe(y_train)