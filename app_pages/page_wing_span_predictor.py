import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your DataFrame with all features
df = pd.read_csv("outputs/datasets/collection/airplane_performance_study.csv")

# Load the trained model
pipeline = joblib.load('outputs/ml_pipeline/predict_analysis/wingspan_predictor_model.pkl')

def page_wing_span_predictor_body():
    """
    Wingspan Predictor App

    This Streamlit application predicts the wingspan of an airplane based on user-inputted features. Users can select values from dropdowns or enter custom values. The app provides warnings for extrapolated values and requires all input fields to be filled before making predictions.

    Features:
    - Input features: Length, Height, AUW, MEW, FW, Vmax, Vcruise, Vstall, Range, Hmax, ROC, Vlo, Slo, Vl, Sl.
    - Validation for complete input before prediction.
    - Extrapolation warnings for out-of-range values.

    Dependencies:
    - streamlit, pandas, numpy, joblib, scikit-learn

    Instructions:
    1. Ensure required libraries are installed.
    2. Load the dataset `airplane_performance_study.csv` and the trained model.
    3. Run the app with `streamlit run your_script.py`.
    """
    st.write("### Wingspan Predictor")

    all_features = [
        'Length', 'Height', 'AUW', 'MEW',
        'FW', 'Vmax', 'Vcruise', 'Vstall', 'Range', 
        'Hmax', 'ROC', 'Vlo', 
        'Slo', 'Vl', 'Sl'
    ]

    input_type = st.radio("Choose input method for all features", 
                           ("Select incremental values from dropdown (interpolation)", 
                            "Enter a custom value (interpolation and extrapolation)"))

    cols = st.columns(3)

    inputs = {}

    for i, feature in enumerate(all_features):
        with cols[i % 3]:
            min_val = df[feature].min()
            max_val = df[feature].max()

            # Format min and max values to 2 decimal places
            min_val_formatted = f"{min_val:.2f}"
            max_val_formatted = f"{max_val:.2f}"

            range_span = max_val - min_val
            remainder = range_span % 1

            min_val_adjusted = min_val + (remainder / 2)
            max_val_adjusted = max_val - (remainder / 2)

            dropdown_values = np.arange(
                np.floor(min_val_adjusted), 
                np.ceil(max_val_adjusted) + 1, 
                1
            ).astype(int)

            if input_type == "Select incremental values from dropdown (interpolation)":
                dropdown_selection = st.selectbox(
                    f"{feature} (data range: {min_val_formatted} - {max_val_formatted})", 
                    options=dropdown_values
                )
                inputs[feature] = dropdown_selection
            elif input_type == "Enter a custom value (interpolation and extrapolation)":
                user_input = st.text_input(f"{feature} (data range: {min_val_formatted} - {max_val_formatted})", "")
                if user_input:
                    try:
                        user_input = float(user_input)
                        inputs[feature] = user_input

                        # Check for extrapolation
                        if user_input < min_val or user_input > max_val:
                            st.warning(f"You have chosen a value outside the data set for {feature}. Extrapolating can be less reliable.")
                    except ValueError:
                        st.error(f"Invalid input for {feature}. Please enter a numerical value.")
                        inputs[feature] = None  # Set to None if input is invalid
                else:
                    inputs[feature] = None  # Set to None if input is empty

    # Collect inputs into a DataFrame
    input_data = pd.DataFrame({
        'Length': [inputs.get('Length', min_val)],
        'Height': [inputs.get('Height', min_val)],
        'AUW': [inputs.get('AUW', min_val)],
        'MEW': [inputs.get('MEW', min_val)],
        'FW': [inputs.get('FW', min_val)],
        'Vmax': [inputs.get('Vmax', min_val)],
        'Vcruise': [inputs.get('Vcruise', min_val)],
        'Vstall': [inputs.get('Vstall', min_val)],
        'Range': [inputs.get('Range', min_val)],
        'Hmax': [inputs.get('Hmax', min_val)],
        'ROC': [inputs.get('ROC', min_val)],
        'Vlo': [inputs.get('Vlo', min_val)],
        'Slo': [inputs.get('Slo', min_val)],
        'Vl': [inputs.get('Vl', min_val)],
        'Sl': [inputs.get('Sl', min_val)]
    })

    # Validation function
    def validate_inputs(inputs):
        # Ensure all inputs are filled
        for feature in all_features:
            if inputs.get(feature) is None:
                return False
        return True

    # Prediction
    if st.button("Predict Wingspan"):
        if validate_inputs(inputs):
            prediction = pipeline.predict(input_data)
            st.write(f"Predicted Wingspan: {prediction[0]:.2f} meters")
            st.write("This prediction is based on the input features provided.")
        else:
            st.error("Please fill in all required features before predicting.")

# Run the predictor function
if __name__ == "__main__":
    page_wing_span_predictor_body()
