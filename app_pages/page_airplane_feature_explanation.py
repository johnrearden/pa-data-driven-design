import streamlit as st
from src.data_management import load_airplane_data, load_pkl_file

def page_airplane_feature_explanation_body():

    st.write("### Explanation of Features")

    # load data
    df = load_airplane_data()

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Features**\n"

        f" **Engine Type** [categorical] Piston, Propjet or Jet type of engine"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Multi Engine** [categorical] Single or multiple engine"
        f" "
        f" "
        f" "
        f" \n"

        f" **TP mods** [categorical] True or False - Most likely: Thrust Performance modifications "
        f" "
        f" "
        f" "
        f" \n"
        
        f" **THR** [lbf] - Thrust for ISA (International Standard Atmosphere)"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **SHP** [HP] - Shaft Horse Power for ISA (International Standard Atmosphere)"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Length** [ft] - Airplane's Length"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Height** [ft] - Airplane's Height"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Wing Span** [ft] - Airplane's Wing Span"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **FW** [lb] - Fuel capacity/weight"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **MEW** [lb] - Empty weight (a.k.a Manufacturer's Empty Weight )"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **AUW** [lb] - Gross weight (a.k.a All-Up Weight)"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Vmax** [knot] - Maximum speed"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Vcruise** [knot] - High cruise speed (Rcmnd cruise)"
        f" "
        f" "
        f" "
        f" \n"

        f" **Vstall** [knot] - Stall speed on 'dirty' configuration (flaps out, gear down, etc.)"
        f" "
        f" "
        f" "
        f" \n"        
        
        f" **Hmax** [ft as density-altitude] - Maximum density-altitude with all engines working"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Hmax (One)** [ft as density-altitude] - Maximum density-altitude with only one engine working."
        f" "
        f" "
        f" "
        f" \n"
        
        f" **ROC** [ft/min] - Rate Of Climb with all engines working"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **ROC (One)** [ft/min] - Rate Of Climb with only one engine working"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Vlo** [ft/min] - Climb speed during normal take-off clearing a 50 ft obstacle"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Slo** [ft] - Takeoff ground run"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Vl** [ft/min] - Landing speed during normal landing clearing a 50 ft obstacle"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Sl** [ft] - Landing ground run"
        f" "
        f" "
        f" "
        f" \n"
        
        f" **Range** [N.m.] The distance the airplane can fly without refueling"
        f" "
        f" "
        f" "
        f" \n"
        

        )

    # inspect data
    if st.checkbox("Inspect Airplane data"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))