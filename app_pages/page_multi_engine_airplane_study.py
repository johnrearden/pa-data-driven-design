import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_airplane_data
import plotly.express as px
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def page_multi_engine_airplane_study_body():

    # load data
    df = load_airplane_data()

    # Hard copied from end of "Using 'Engine Type' as target variable" in "3A_airplane_engine_type_study"-notebook.
    vars_to_study = ['Hmax', 'Vcruise', 'Vl', 'Vmax', 'Vstall']

    st.write("### Multi Engine Airplane Study")
    st.info(
        f"* The client is interested in understanding how Multi Engined airplanes"
        f" differ in terms of Performance to that of single Engined Airplanes so that"
        f" he can make better selections in the Conceptual Design Phase.")

    # inspect data
    if st.checkbox("Inspect Airplane data"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))

    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in the notebook to better understand how "
        f"**Multi Engine** is correlated to the other features. \n"
        f"The most correlated features are: **{vars_to_study}**"
    )

    # Text based on "3A_airplane_engine_type_study" notebook - "Conclusions"-section
    st.info(
        f"The correlation and plots (Multi Engine distributed more to the right in the histograms"
        f" and at higher values in parallel plot) below clearly shows that a"
        f" Multi Engined Airplane on average has a **higher: Hmax, Vcruise, Vl, Vmax and Vstall**"
        f" than a Single Engined Airplane. \n"
    )

    # Code copied from "3A_airplane_engine_type_study" notebook - "EDA on selected variables"-section
    df_eda = df.filter(vars_to_study + ['Multi_Engine'])

    # Individual plots per variable
    if st.checkbox("Multi Engine distribution per Feature"):
        multi_engine_per_feature(df_eda, vars_to_study)

    # Parallel plot
    if st.checkbox("Parallel Plot"):
        st.write(
            f"* Information in yellow shows the profile from a Multi Engined Airplane")
        parallel_plot_multi_engine(df_eda)


# Code copied from "3A_airplane_engine_type_study" notebook - "Variables Distribution by Multi Engine"-section
def plot_numerical(df, col, target_var):
    fig, ax = plt.subplots(figsize=(8, 5))  # Create a figure and axis
    sns.histplot(data=df, x=col, hue=target_var, kde=True, element="step", ax=ax)  # Pass ax to the plot
    ax.set_title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)  # Pass the figure to st.pyplot()


# Code copied from "3A_airplane_engine_type_study" notebook - "Variables Distribution by Multi Engine"-section
def multi_engine_per_feature(df_eda, vars_to_study):
    target_var = 'Multi_Engine'
    for col in vars_to_study:
        plot_numerical(df_eda, col, target_var)
        print("\n\n")


# Function created using "3A_airplane_engine_type_study" notebook code - "Parallel Plot"-section
def parallel_plot_multi_engine(df_eda):

    # Define the mapping arrays
    # Maps hard coded based on inspection of the histogram plots under "Variables Distribution by Multi Engine" in 3A_airplane_engine_type_study.
    Hmax_map = [-np.Inf, 23000, 32000, 42000, 50000, np.Inf]
    Vcruise_map = [-np.Inf, 250, 350, 450, 550, np.Inf]
    Vl_map = [-np.Inf, 2000, 3000, 4000, np.Inf]
    Vmax_map = [-np.Inf, 250, 350, 450, 550, np.Inf]
    Vstall_map = [-np.Inf, 70, 90, 110, np.Inf]

    # Combine all mappings into a single binning dictionary (Inbetween step necessary since we have multiple variables)
    binning_dict = {
        'Hmax': Hmax_map,
        'Vcruise': Vcruise_map,
        'Vl': Vl_map,
        'Vmax': Vmax_map,
        'Vstall': Vstall_map
    }

    # Initialize the ArbitraryDiscretiser with the combined binning dictionary
    disc = ArbitraryDiscretiser(binning_dict=binning_dict)

    # Fit and transform the DataFrame
    df_parallel = disc.fit_transform(df_eda)

    # Access the binning dictionaries after fitting
    if hasattr(disc, 'binner_dict_'):
        print("Binning dictionary for Hmax:", disc.binner_dict_['Hmax'])
        print("Binning dictionary for Vcruise:", disc.binner_dict_['Vcruise'])
        print("Binning dictionary for Vl:", disc.binner_dict_['Vl'])
        print("Binning dictionary for Vmax:", disc.binner_dict_['Vmax'])
        print("Binning dictionary for Vstall:", disc.binner_dict_['Vstall'])
    else:
        print("binner_dict_ does not exist. Please check if the discretiser was fitted successfully.")

    # Unsure if I need this line CHECK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    labels_map = {}

    # Iterate over each variable in the binning dictionary
    for variable in disc.binner_dict_.keys():
        classes_ranges = disc.binner_dict_[variable][1:-1]  # Exclude -Inf and +Inf
        n_classes = len(classes_ranges) + 1  # Number of intervals/classes
    
        # Initialize labels for this variable
        variable_labels = {}
    
        for n in range(n_classes):
            if n == 0:
                variable_labels[n] = f"<{classes_ranges[0]}"
            elif n == n_classes - 1:
                variable_labels[n] = f"+{classes_ranges[-1]}"
            else:
                variable_labels[n] = f"{classes_ranges[n - 1]} to {classes_ranges[n]}"
    
        # Store the labels in the main labels_map
        labels_map[variable] = variable_labels

    # Replace the values in df_parallel for each variable using the corresponding labels from labels_map
    for variable, labels in labels_map.items():
        df_parallel[variable] = df_parallel[variable].replace(labels)

    # Convert boolean to integer via replacing
    df_parallel['Multi_Engine'] = df_parallel['Multi_Engine'].replace({True: 1, False: 0})

    # Reverse the order of the categories for each variable
    for variable in labels_map.keys():
        df_parallel[variable] = pd.Categorical(df_parallel[variable], categories=sorted(labels_map[variable].values(), reverse=True), ordered=True)

    fig = px.parallel_categories(df_parallel, color="Multi_Engine")   #fig = px.parallel_categories(df_parallel, color="Multi_Engine", color_discrete_sequence=["blue", "orange"])
    #fig.show(renderer='jupyterlab')
    # we use st.plotly_chart() to render, in notebook is fig.show()
    st.plotly_chart(fig)
