import streamlit as st

from src.data_management import load_airplane_data, load_pkl_file


def page_airplane_feature_explanation_body():
    st.write("### Explanation of Features")

    # load data
    df = load_airplane_data()

    # text based on README file - "Dataset Content" section
    st.info(

        f" **Engine Type** [categorical] Piston, Propjet or Jet type of engine  \n"

        f" **Multi Engine** [categorical] Single or multiple engine  \n"

        f" **TP mods** [categorical] True or False - Most likely: Thrust Performance modifications  \n"

        f" **THR** [lbf] - Thrust for ISA (International Standard Atmosphere)  \n"

        f" **SHP** [HP] - Shaft Horse Power for ISA (International Standard Atmosphere)  \n"

        f" **Length** [ft] - Airplane's Length  \n"

        f" **Height** [ft] - Airplane's Height  \n"

        f" **Wing Span** [ft] - Airplane's Wing Span  \n"

        f" **FW** [lb] - Fuel capacity/weight  \n"

        f" **MEW** [lb] - Empty weight (a.k.a Manufacturer's Empty Weight )  \n"

        f" **AUW** [lb] - Gross weight (a.k.a All-Up Weight)  \n"

        f" **Vmax** [knot] - Maximum speed  \n"

        f" **Vcruise** [knot] - High cruise speed (Rcmnd cruise)  \n"

        f" **Vstall** [knot] - Stall speed on 'dirty' configuration (flaps out, gear down, etc.)  \n"

        f" **Hmax** [ft as density-altitude] - Maximum density-altitude with all engines working  \n"

        f" **Hmax (One)** [ft as density-altitude] - Maximum density-altitude with only one engine working.  \n"

        f" **ROC** [ft/min] - Rate Of Climb with all engines working  \n"

        f" **ROC (One)** [ft/min] - Rate Of Climb with only one engine working  \n"

        f" **Vlo** [ft/min] - Climb speed during normal take-off clearing a 50 ft obstacle  \n"

        f" **Slo** [ft] - Takeoff ground run  \n"

        f" **Vl** [ft/min] - Landing speed during normal landing clearing a 50 ft obstacle  \n"

        f" **Sl** [ft] - Landing ground run  \n"

        f" **Range** [N.m.] The distance the airplane can fly without refueling  \n"
        )

    # inspect data
    if st.checkbox("Inspect Airplane data"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))
