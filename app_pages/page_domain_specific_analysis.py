import plotly.express as px
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
sns.set_style("whitegrid")


def page_domain_specific_analysis_body():
    # Load data
    df = pd.read_csv('outputs/datasets/collection/airplane_performance_study.csv')

    print(df)

    st.write("### Domain Specific Studies")

    st.write(
        f"* Two simple correlation studies were conducted to showcase interesting relationships"
        f" between design features and performance features. \n"
    )
    st.write("---")
    
    st.info(
        f"The late Prof. Peter Lissaman, who developed the solar-powered High Altitude Airplane Helios"
        f" (Aerovironment/NASA), found that Hmax was strongly dependent on wing span:"
        f" as wing span increases, Hmax also increases."
        f" However this relationship is not immediately obvious from the equations, and the airplanes in the" 
        f" dataset differ vastly from the data set:"
        f" Helios was ultra-light, propeller-driven with 14 electric engines and it operated"
        f" in a completely different speed and altitude regime."
        f" Despite this, we can see this relationship for the Piston (propeller-driven) Engine Type.\n"
    )

    # Individual plots per variable
    if st.checkbox("View Effect of Wing Span on Hmax by Engine Type plot"):
        ceiling_as_function_of_wingspan(df)

    st.write("---")


    st.info(
        """
        Let's put the Breguet Range equation to the test with some real-world data! 
        The equation features the ratio between the airplane's weight at the start (with fuel) 
        and weight at landing (after the fuel has been burned off). It stands to reason that 
        the distance an airplane can fly is directly related to how much of its weight consists of fuel.

        The data with its steep sloped positive regression line gives Breguet credit; however, 
        the data also hint that the relationship is only close to linear yet not completely linear! 
        More analysis would be needed.

        **Breguet Range Equation:**
        R = (η_pr / c) × (L/D) × (W₀ / W₁)

        Where:
        R = range (distance the aircraft can travel)  
        η_pr = propulsive efficiency  
        c = specific fuel consumption  
        L/D = lift-to-drag ratio  
        W₀ = initial weight of the aircraft (including fuel)  
        W₁ = final weight of the aircraft (after fuel is burned)  
        """
    )


    #st.markdown(r"""
    ### Breguet Range Equation
    #$$
    #R = \frac{\eta_{pr}}{c} \times \frac{L}{D} \times \frac{W_0}{W_1}
    #$$
    #Where:
    #- \( R \) = range (distance the aircraft can travel)
    #- \( \eta_{pr} \) = propulsive efficiency
    #- \( c \) = specific fuel consumption
    #- \( \frac{L}{D} \) = lift-to-drag ratio
    #- \( W_0 \) = initial weight of the aircraft (including fuel)
    #- \( W_1 \) = final weight of the aircraft (after fuel is burned)
    #""")


    # Plot: range_as_function_of_weight_ratio
    if st.checkbox("View Effect of Wi/Wf on Range plot"):
        range_as_function_of_weight_ratio(df)

def ceiling_as_function_of_wingspan(df):
    # Set the style for the plot
    sns.set(style="whitegrid")

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot regression lines for different Engine Types
    sns.regplot(x='Wing_Span', y='Hmax', data=df[df['Engine_Type'] == 0], 
                scatter_kws={'alpha': 0.5}, 
                line_kws={'label': 'Piston'}, 
                color='blue')
    sns.regplot(x='Wing_Span', y='Hmax', data=df[df['Engine_Type'] == 1], 
                scatter_kws={'alpha': 0.5}, 
                line_kws={'label': 'Propjet'}, 
                color='orange')
    sns.regplot(x='Wing_Span', y='Hmax', data=df[df['Engine_Type'] == 2], 
                scatter_kws={'alpha': 0.5}, 
                line_kws={'label': 'Jet'}, 
                color='green')

    # Add legend and titles
    plt.legend()
    plt.title('Effect of Wing Span on Hmax by Engine Type')
    plt.xlabel('Wing Span')
    plt.ylabel('Hmax')

    # Display the plot in Streamlit
    st.pyplot(plt)  # Use st.pyplot instead of plt.show()

def range_as_function_of_weight_ratio(df):
    # Remove extreme outlier from the DataFrame
    df = df.drop(index=467).reset_index(drop=True)

    # Calculate Wi / Wf based on the formula
    df['Wi/Wf'] = df['AUW'] / (df['AUW'] - df['FW'])

    # Create a regression plot using seaborn
    plt.figure(figsize=(10, 6))

    # Linear regression with orange scatter points
    sns.regplot(x='Wi/Wf', y='Range', data=df, label='Linear Regression', color='blue', 
                scatter_kws={'color': 'orange'})

    # Quadratic regression
    sns.regplot(x='Wi/Wf', y='Range', data=df, order=2, label='Quadratic Regression', color='red', 
                scatter_kws={'color': 'orange'})  # Keep scatter orange for both

    # Customize the plot
    plt.title('Effect of Wi/Wf on Range')
    plt.xlabel('Wi/Wf')
    plt.ylabel('Range')
    plt.grid()

    # Create custom legend handles
    custom_legend = [
        Line2D([0], [0], color='blue', label='Linear Regression'),
        Line2D([0], [0], color='red', label='Quadratic Regression')
    ]
    plt.legend(handles=custom_legend)

    # Display the plot in Streamlit
    st.pyplot(plt)
