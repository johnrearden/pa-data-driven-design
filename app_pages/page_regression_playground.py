import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression

def airplane_performance_study():
    path = '/workspace/data-driven-design/outputs/datasets/collection/airplane_performance_study.csv'
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()  # Strip whitespace from column names
    return df

def page_regression_playground_body():
    st.write("### Piper vs. Cessna")
    st.info("* We have pitched Piper against Cessna ...")
    st.write("---")

    X_live = DrawInputsWidgets()

    if st.button("Create Regression Plot"):
        dependent_feature = X_live["Dependent feature"].values[0]
        independent_feature_1 = X_live["Independent feature 1"].values[0]
        independent_feature_2 = X_live["Independent feature 2"].values[0]
        filter_option = X_live["Filter Option"].values[0]

        df = airplane_performance_study()  # Load the DataFrame

        if filter_option == "Piper vs. Cessna":
            df = df[df['Company'].isin(['Piper Aircraft', 'Cessna Aircraft Company'])]

        graph_type = X_live["Type of graph"].values[0]
        if graph_type == "2D Regression":
            plot_2d_regression(df, dependent_feature, independent_feature_1, independent_feature_2, filter_option)
        elif graph_type == "3D Regression":
            plot_3d_regression(df, dependent_feature, independent_feature_1, independent_feature_2, filter_option)
        else:
            st.error("Please select valid features.")

def DrawInputsWidgets():
    df = airplane_performance_study()
    excluded_features = ["Multi_Engine", "TP_mods", "Engine_Type", "Model", "Company"]  # Exclude 'Company' for feature selection
    available_features = [col for col in df.columns if col not in excluded_features]

    col1, col2, col3 = st.columns(3)
    X_live = pd.DataFrame([], index=[0])

    with col1:
        feature = "Dependent feature"
        st_widget = st.selectbox(label=feature, options=available_features, index=available_features.index("Vmax"))
    X_live[feature] = st_widget

    with col2:
        feature = "Independent feature 1"
        st_widget = st.selectbox(label=feature, options=available_features, index=available_features.index("Wing_Span"))
    X_live[feature] = st_widget

    with col3:
        feature = "Independent feature 2"
        st_widget = st.selectbox(label=feature, options=available_features, index=available_features.index("AUW"))
    X_live[feature] = st_widget

    col4, col5 = st.columns([1, 1])
    with col4:
        feature = "Filter Option"
        st_widget = st.selectbox(label=feature, options=["All Airplanes", "Piper vs. Cessna"])
    X_live[feature] = st_widget

    with col5:
        feature = "Type of graph"
        st_widget = st.selectbox(label=feature, options=["2D Regression", "3D Regression"])
    X_live[feature] = st_widget

    return X_live

def plot_2d_regression(df, dependent_feature, independent_feature_1, independent_feature_2, filter_option):
    filtered_df = df[[dependent_feature, independent_feature_1, independent_feature_2, 'Company']].dropna()

    plt.figure(figsize=(10, 5))
    if filter_option == "Piper vs. Cessna":
        for company in ['Piper Aircraft', 'Cessna Aircraft Company']:
            company_df = filtered_df[filtered_df['Company'] == company]
            sns.regplot(data=company_df, x=independent_feature_1, y=dependent_feature, label=company)
        plt.legend()  # Show legend for Piper vs. Cessna
    else:
        sns.regplot(data=filtered_df, x=independent_feature_1, y=dependent_feature)

    plt.title(f'{dependent_feature} vs. {independent_feature_1}')
    plt.xlabel(independent_feature_1)
    plt.ylabel(dependent_feature)
    plt.grid()
    st.pyplot(plt)

    plt.figure(figsize=(10, 5))
    if filter_option == "Piper vs. Cessna":
        for company in ['Piper Aircraft', 'Cessna Aircraft Company']:
            company_df = filtered_df[filtered_df['Company'] == company]
            sns.regplot(data=company_df, x=independent_feature_2, y=dependent_feature, label=company)
        plt.legend()  # Show legend for Piper vs. Cessna
    else:
        sns.regplot(data=filtered_df, x=independent_feature_2, y=dependent_feature)

    plt.title(f'{dependent_feature} vs. {independent_feature_2}')
    plt.xlabel(independent_feature_2)
    plt.ylabel(dependent_feature)
    plt.grid()
    st.pyplot(plt)

def plot_3d_regression(df, dependent_feature, independent_feature_1, independent_feature_2, filter_option):
    filtered_df = df[[dependent_feature, independent_feature_1, independent_feature_2, 'Company']].dropna()

    fig = go.Figure()
    if filter_option == "Piper vs. Cessna":
        for company in ['Piper Aircraft', 'Cessna Aircraft Company']:
            company_df = filtered_df[filtered_df['Company'] == company]
            X = company_df[[independent_feature_1, independent_feature_2]]
            y = company_df[dependent_feature]

            model = LinearRegression()
            model.fit(X, y)

            x_range = np.linspace(X[independent_feature_1].min(), X[independent_feature_1].max(), 100)
            y_range = np.linspace(X[independent_feature_2].min(), X[independent_feature_2].max(), 100)
            x_grid, y_grid = np.meshgrid(x_range, y_range)
            Z = model.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)

            fig.add_trace(go.Surface(z=Z, x=x_grid, y=y_grid, colorscale='Viridis', opacity=0.5, name=f'{company} Regression Plane'))
            fig.add_trace(go.Scatter3d(x=X[independent_feature_1], y=X[independent_feature_2], z=y, mode='markers',
                                         marker=dict(size=5, opacity=0.8), name=f'{company} Data Points'))
    else:
        X = filtered_df[[independent_feature_1, independent_feature_2]]
        y = filtered_df[dependent_feature]

        model = LinearRegression()
        model.fit(X, y)

        x_range = np.linspace(X[independent_feature_1].min(), X[independent_feature_1].max(), 100)
        y_range = np.linspace(X[independent_feature_2].min(), X[independent_feature_2].max(), 100)
        x_grid, y_grid = np.meshgrid(x_range, y_range)
        Z = model.predict(np.c_[x_grid.ravel(), y_grid.ravel()]).reshape(x_grid.shape)

        fig.add_trace(go.Surface(z=Z, x=x_grid, y=y_grid, colorscale='Viridis', opacity=0.5, name='All Airplanes Regression Plane'))
        fig.add_trace(go.Scatter3d(x=X[independent_feature_1], y=X[independent_feature_2], z=y, mode='markers',
                                     marker=dict(size=5, opacity=0.8), name='All Airplanes Data Points'))

    fig.update_layout(title='3D Regression Plane with Scatter Plot',
                      scene=dict(xaxis_title=independent_feature_1,
                                 yaxis_title=independent_feature_2,
                                 zaxis_title=dependent_feature),
                      autosize=True)

    st.plotly_chart(fig)

if __name__ == "__main__":
    page_piper_vs_cessna_body()
