import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def airplane_performance_study():
    """
    Load the airplane performance dataset from a CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing the airplane performance data
        with whitespace stripped from column names.
    """
    path = 'outputs/datasets/collection/airplane_performance_study.csv'
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # Strip whitespace from column names
    return df


def page_regression_playground_body():
    """
    Display the main body of the Regression Playground page in Streamlit.

    This function sets up the user interface for selecting features,
    filters, and regression types, and calls the appropriate plotting
    functions based on user input.
    """
    st.write("### Regression Playground")
    st.info("* All the continuous numeric features are available for you to explore in the menu below!"
            " The shaded background around the regression line shows variance (narrow variance"
            " indicate higher predictive reliability.  \n"  
            " For fun we have pitched Piper and Cessna head to head against each other to see who's"
            " regression lines/surfaces come out on top in the different disciplines! "
            " Note that the true merit of an airplane is not captured by only comparing a limited number of features.\n")


    X_live = DrawInputsWidgets()
    if st.button("Create Regression Plot"):
        dependent_feature = X_live["Dependent feature"].values[0]
        independent_feature_1 = X_live["Independent feature 1"].values[0]
        independent_feature_2 = X_live["Independent feature 2"].values[0]
        filter_option = X_live["Filter Option"].values[0]
        regression_type = X_live["Regression Type"].values[0]

        df = airplane_performance_study()  # Load the DataFrame

        if filter_option == "Piper vs. Cessna":
            df = df[df['Company'].isin(['Piper Aircraft', 'Cessna Aircraft Company'])]

        # Error handling when dataFrame is empty (mainly case for Piper vs. Cessna)
        if df.empty:
            st.error("No data available after filtering. Please adjust your filters.")
            return

        graph_type = X_live["Type of graph"].values[0]
        if graph_type == "2D Regression":
            plot_2d_regression(df, dependent_feature, independent_feature_1, independent_feature_2, filter_option, regression_type)
        elif graph_type == "3D Regression":
            plot_3d_regression(df, dependent_feature, independent_feature_1, independent_feature_2, filter_option, regression_type)
        else:
            st.error("Please select valid features.")


def DrawInputsWidgets():
    """
    Create input widgets for selecting features and options for regression analysis.

    Returns:
        pd.DataFrame: A DataFrame containing the selected features and options
        from the user input.
    """
    df = airplane_performance_study()
    excluded_features = ["Multi_Engine", "TP_mods", "Engine_Type", "Model", "Company"]  # Exclude 'Company' for feature selection
    available_features = [col for col in df.columns if col not in excluded_features]

    col1, col2, col3 = st.columns(3)
    X_live = pd.DataFrame([], index=[0])

    with col1:
        feature = "Dependent feature"
        st.session_state.dependent_feature = st.selectbox(label=feature, options=available_features, index=0)  # Set the Wing Span as default
    X_live[feature] = st.session_state.dependent_feature

    with col2:
        feature = "Independent feature 1"
        st.session_state.independent_feature_1 = st.selectbox(label=feature, options=available_features, index=5)  # Set AUW as default
    X_live[feature] = st.session_state.independent_feature_1

    with col3:
        feature = "Independent feature 2"
        st.session_state.independent_feature_2 = st.selectbox(label=feature, options=available_features, index=11)  # Set Range as default
    X_live[feature] = st.session_state.independent_feature_2

    # Check for duplicate selections (in the top row of selection dropdowns)
    if (st.session_state.independent_feature_1 == st.session_state.dependent_feature or
        st.session_state.independent_feature_2 == st.session_state.dependent_feature or
        st.session_state.independent_feature_1 == st.session_state.independent_feature_2):
        st.warning("Ensure that all selected features are different.")

    # New row for Filter Option, Regression Type, and Type of Graph
    col4, col5, col6 = st.columns(3)
    with col4:
        feature = "Filter Option"
        st_widget = st.selectbox(label=feature, options=["All Airplanes", "Piper vs. Cessna"])
    X_live[feature] = st_widget

    with col5:
        feature = "Regression Type"
        st_widget = st.selectbox(label=feature, options=["Linear", "Quadratic"])
    X_live[feature] = st_widget

    with col6:
        feature = "Type of graph"
        st_widget = st.selectbox(label=feature, options=["2D Regression", "3D Regression"])
    X_live[feature] = st_widget

    return X_live


def plot_2d_regression(df, dependent_feature, independent_feature_1, independent_feature_2, filter_option, regression_type):
    """
    Plot 2D regression using seaborn based on selected features and regression type.

    Args:
        df (pd.DataFrame): The DataFrame containing airplane performance data.
        dependent_feature (str): The dependent feature for the regression.
        independent_feature_1 (str): The first independent feature.
        independent_feature_2 (str): The second independent feature.
        filter_option (str): The filter option selected by the user.
        regression_type (str): The type of regression ('Linear' or 'Quadratic').
    """
    filtered_df = df[[dependent_feature, independent_feature_1, independent_feature_2, 'Company']].dropna()

    if filtered_df.empty:
        st.error("No data available for the selected features.")
        return

    plt.figure(figsize=(10, 5))
    if filter_option == "Piper vs. Cessna":
        for company in ['Piper Aircraft', 'Cessna Aircraft Company']:
            company_df = filtered_df[filtered_df['Company'] == company]
            if regression_type == "Linear":
                sns.regplot(data=company_df, x=independent_feature_1, y=dependent_feature, label=company)
            elif regression_type == "Quadratic":
                sns.regplot(data=company_df, x=independent_feature_1, y=dependent_feature, label=company, order=2)
        plt.legend()
    else:
        if regression_type == "Linear":
            sns.regplot(data=filtered_df, x=independent_feature_1, y=dependent_feature)
        elif regression_type == "Quadratic":
            sns.regplot(data=filtered_df, x=independent_feature_1, y=dependent_feature, order=2)

    plt.title(f'{dependent_feature} vs. {independent_feature_1}')
    plt.xlabel(independent_feature_1)
    plt.ylabel(dependent_feature)
    plt.grid()
    st.pyplot(plt)

    plt.figure(figsize=(10, 5))
    if filter_option == "Piper vs. Cessna":
        for company in ['Piper Aircraft', 'Cessna Aircraft Company']:
            company_df = filtered_df[filtered_df['Company'] == company]
            if regression_type == "Linear":
                sns.regplot(data=company_df, x=independent_feature_2, y=dependent_feature, label=company)
            elif regression_type == "Quadratic":
                sns.regplot(data=company_df, x=independent_feature_2, y=dependent_feature, label=company, order=2)
        plt.legend()
    else:
        if regression_type == "Linear":
            sns.regplot(data=filtered_df, x=independent_feature_2, y=dependent_feature)
        elif regression_type == "Quadratic":
            sns.regplot(data=filtered_df, x=independent_feature_2, y=dependent_feature, order=2)

    plt.title(f'{dependent_feature} vs. {independent_feature_2}')
    plt.xlabel(independent_feature_2)
    plt.ylabel(dependent_feature)
    plt.grid()
    st.pyplot(plt)


def plot_3d_regression(df, dependent_feature, independent_feature_1, independent_feature_2, filter_option, regression_type):
    """
    Plot 3D regression using Plotly based on selected features and regression type.

    Args:
        df (pd.DataFrame): The DataFrame containing airplane performance data.
        dependent_feature (str): The dependent feature for the regression.
        independent_feature_1 (str): The first independent feature.
        independent_feature_2 (str): The second independent feature.
        filter_option (str): The filter option selected by the user.
        regression_type (str): The type of regression ('Linear' or 'Quadratic').
    """
    filtered_df = df[[dependent_feature, independent_feature_1, independent_feature_2, 'Company']].dropna()

    # Check if there is data available after filtering (part of teh error handling)
    if filtered_df.empty or filtered_df.shape[0] < 2:
        st.error("Not enough data available for the selected features to perform regression.")
        return

    fig = go.Figure()
    if filter_option == "Piper vs. Cessna":
        for company in ['Piper Aircraft', 'Cessna Aircraft Company']:
            company_df = filtered_df[filtered_df['Company'] == company]

            # Ensure there's enough data for fitting the model
            if company_df.shape[0] < 2:
                st.warning(f"Not enough data for {company}. Skipping this Company.")
                continue

            X = company_df[[independent_feature_1, independent_feature_2]]
            y = company_df[dependent_feature]

            if regression_type == "Linear":
                model = LinearRegression()
                model.fit(X, y)
                x_range = np.linspace(X[independent_feature_1].min(), X[independent_feature_1].max(), 100)
                y_range = np.linspace(X[independent_feature_2].min(), X[independent_feature_2].max(), 100)
                x_grid, y_grid = np.meshgrid(x_range, y_range)
                Z = model.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)

            elif regression_type == "Quadratic":
                poly = PolynomialFeatures(degree=2)
                X_poly = poly.fit_transform(X)
                model = LinearRegression()
                model.fit(X_poly, y)
                x_range = np.linspace(X[independent_feature_1].min(), X[independent_feature_1].max(), 100)
                y_range = np.linspace(X[independent_feature_2].min(), X[independent_feature_2].max(), 100)
                x_grid, y_grid = np.meshgrid(x_range, y_range)
                X_grid = poly.transform(np.c_[x_grid.ravel(), y_grid.ravel()])
                Z = model.predict(X_grid).reshape(x_grid.shape)

            # Add the surface and scatter plots to the figure
            fig.add_trace(go.Surface(z=Z, x=x_grid, y=y_grid, colorscale='Viridis', opacity=0.5, name=f'{company} Regression Plane'))
            fig.add_trace(go.Scatter3d(x=X[independent_feature_1], y=X[independent_feature_2], z=y, mode='markers',
                                       marker=dict(size=5, opacity=0.8), name=f'{company} Data Points'))
    else:
        X = filtered_df[[independent_feature_1, independent_feature_2]]
        y = filtered_df[dependent_feature]

        # Ensure there's enough data for fitting the model
        if X.shape[0] < 2:
            st.error("Not enough data available for the selected features to perform regression.")
            return

        if regression_type == "Linear":
            model = LinearRegression()
            model.fit(X, y)
            x_range = np.linspace(X[independent_feature_1].min(), X[independent_feature_1].max(), 100)
            y_range = np.linspace(X[independent_feature_2].min(), X[independent_feature_2].max(), 100)
            x_grid, y_grid = np.meshgrid(x_range, y_range)
            Z = model.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)

        elif regression_type == "Quadratic":
            poly = PolynomialFeatures(degree=2)
            X_poly = poly.fit_transform(X)
            model = LinearRegression()
            model.fit(X_poly, y)
            x_range = np.linspace(X[independent_feature_1].min(), X[independent_feature_1].max(), 100)
            y_range = np.linspace(X[independent_feature_2].min(), X[independent_feature_2].max(), 100)
            x_grid, y_grid = np.meshgrid(x_range, y_range)
            X_grid = poly.transform(np.c_[x_grid.ravel(), y_grid.ravel()])
            Z = model.predict(X_grid).reshape(x_grid.shape)

        fig.add_trace(go.Surface(z=Z, x=x_grid, y=y_grid, colorscale='Viridis', opacity=0.5, name='All Airplanes Regression Plane'))
        fig.add_trace(go.Scatter3d(x=X[independent_feature_1], y=X[independent_feature_2], z=y, mode='markers',
                                   marker=dict(size=5, opacity=0.8), name='All Airplanes Data Points'))

    fig.update_layout(title='3D Regression Plane with Scatter Plot',
                      scene=dict(xaxis_title=independent_feature_1,
                                 yaxis_title=independent_feature_2,
                                 zaxis_title=dependent_feature),
                      autosize=True)

    st.plotly_chart(fig)
