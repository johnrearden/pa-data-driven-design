import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load DataFrame with all features
df = pd.read_csv("outputs/datasets/collection/airplane_performance_study.csv")

# Load the trained model
pipeline = joblib.load('outputs/ml_pipeline/predict_analysis/wingspan_predictor_model.pkl')

def page_wing_span_predictor_body():
    """
    Wingspan Predictor App
    """

    st.write("### Wingspan Predictor")
    st.info(
        f"The Wing Span predictor computes what wing span an airplane need to have in order to"
        f" reach the performance target that you set below."
        f" Each features mean value is set as default."
    )

    st.write("---")
    
    # List of all feature names
    all_features = [
        'Length', 'Height', 'AUW', 'MEW',
        'FW', 'Vmax', 'Vcruise', 'Vstall', 'Range', 
        'Hmax', 'ROC', 'Vlo', 
        'Slo', 'Vl', 'Sl'
    ]

    # Define units for each feature in a dictionary
    units = {
        'Length': 'ft', 
        'Height': 'ft', 
        'AUW': 'lb', 
        'MEW': 'lb', 
        'FW': 'lb', 
        'Vmax': 'knots', 
        'Vcruise': 'knots', 
        'Vstall': 'knots', 
        'Range': 'N.m.', 
        'Hmax': 'ft', 
        'ROC': 'ft/min', 
        'Vlo': 'ft/min', 
        'Slo': 'ft', 
        'Vl': 'ft/min', 
        'Sl': 'ft'
    }

    # Calculate the rounded mean values for all features
    mean_values = {feature: round(df[feature].mean()) for feature in all_features}

    input_type = st.radio("",  # "Choose input method for all features"
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
            max_val_adjusted = max_val - (remainder / 2)  # Ensure we use the correct variable name

            # Create dropdown values as regular Python integers
            dropdown_values = list(range(int(np.floor(min_val_adjusted)), int(np.ceil(max_val_adjusted)) + 1))

            # If using dropdown, default to the rounded mean value
            if input_type == "Select incremental values from dropdown (interpolation)":
                # Select the index of the closest value to the rounded mean
                closest_value_index = min(range(len(dropdown_values)), key=lambda i: abs(dropdown_values[i] - mean_values[feature]))
                dropdown_selection = st.selectbox(
                    f"{feature} [{units[feature]}] (data range: {min_val_formatted} - {max_val_formatted})", 
                    options=dropdown_values,  # Use list of integers
                    index=closest_value_index  # Set the mean value as default
                )
                inputs[feature] = dropdown_selection

            # If using text input, default to the rounded mean value
            elif input_type == "Enter a custom value (interpolation and extrapolation)":
                user_input = st.text_input(f"{feature} [{units[feature]}] (data range: {min_val_formatted} - {max_val_formatted})", "")
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
                    # Use the rounded mean value as default if input is empty
                    inputs[feature] = mean_values[feature]  # Set to rounded mean value as default

    # Collect inputs into a DataFrame
    input_data = pd.DataFrame({
        'Length': [inputs.get('Length', mean_values['Length'])],
        'Height': [inputs.get('Height', mean_values['Height'])],
        'AUW': [inputs.get('AUW', mean_values['AUW'])],
        'MEW': [inputs.get('MEW', mean_values['MEW'])],
        'FW': [inputs.get('FW', mean_values['FW'])],
        'Vmax': [inputs.get('Vmax', mean_values['Vmax'])],
        'Vcruise': [inputs.get('Vcruise', mean_values['Vcruise'])],
        'Vstall': [inputs.get('Vstall', mean_values['Vstall'])],
        'Range': [inputs.get('Range', mean_values['Range'])],
        'Hmax': [inputs.get('Hmax', mean_values['Hmax'])],
        'ROC': [inputs.get('ROC', mean_values['ROC'])],
        'Vlo': [inputs.get('Vlo', mean_values['Vlo'])],
        'Slo': [inputs.get('Slo', mean_values['Slo'])],
        'Vl': [inputs.get('Vl', mean_values['Vl'])],
        'Sl': [inputs.get('Sl', mean_values['Sl'])]
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
            st.write(f"Predicted Wingspan: {prediction[0]:.2f} ft")
            st.write("This prediction is based on the input features provided.")
        else:
            st.error("Please fill in all required features before predicting.")

# Run the predictor function
if __name__ == "__main__":
    page_wing_span_predictor_body()