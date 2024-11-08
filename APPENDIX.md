# APPENDIX 

## Buisness case for unfinished Multi Engine Classification Model
- We want an ML model to predict if an Airplane have a Multi Engine or single Engine on historical General Aviation Airplane data. The target variable is categorical and contains 2-classes. We consider a **classification model**. It is a supervised model, a 2-class, single-label, classification model output: 0 (no Multi Engine), 1 (yes Multi Engine).
- Our ideal outcome is to provide our client with a predictor tool that can assist in feasibility studies of new proposal.
- The model success metrics are
	- at least 80% Recall for Multi Engine, on train and test set 
	- The ML model is considered a failure if:
		- Precision for no Multi Engine is lower than 80% on train and test set. (We don't want wrong decisions on configuration, such as Single or Multi Engine to be passed on to the Preliminary design phase potentially resulting in extremely costly redesigns if the misstake  isn ot discovered until the Preliminary Design phase or, even worse, in the Detailed Design phase)
- The model output is defined as a flag, indicating if an Airplane have Multi or Single Engines.
- Heuristics: Currently, there are many predictor tools similar to this however most are in-house (and therefore not accessible) and our client want to develope a solid base of in-house tools to predict Design parameters such as if an airplane is more suited to have Single or Multi Engines.
- The training data to fit the model comes from the Kaggle data set.
	- Train data - features: all variables, but: Model, Company, THR, SHP.

## Units of data set features
**Page Engine Type**
Note that a conversion to SI units has not been made in the data set analysis.

|      Quantity     | Meaning/Information/Quantity | Data set units (traditional Aviation units) | SI units |
|-------------------|------------------------------|---------------------------------------------|----------|
| "Propulsion size" | THR, SHP            | lbf and HP                                  | N and W  |   
| Length            | Wing Span, Lenght, Height, Slo and Sl           | ft and in                                   |    m     |  
| Distance          | Range                        | N.m. (Nautical miles)                       |   km     |  
| Weight            | FW, AUW and MEW            | lb                                          | kg or N  |
| Velocity          | Vmax, Vcruise, Vstall,          | knot or Mach and in                         |   m/s    |   
| Vertical velocity | ROC, Vlo and Vl          | ft/min                                      |   m/s    |

## Domain specific comments on relationships between the features in the data set
Outlined below are the dependencies between the features in the data set (and features mentioned in the Outlook) relevant for making hypotheses. Other dataset features are encircled in red as they appear in the equations. Underlined features indicate that they are indirectly related to other features in the dataset however the selection of which features to underline is rather ambiguous.

### Engine Type (categories: Piston and propjet, jet)
Jet generally offers higher **speed** and **ceilings** as well as better **range**. Propjet generally falls somewhere between these two engine types.  Piston powered propeller driven propulsion units meets an invisible "speed barrier" approaching 400 knots. One reasons for this "barrier" is because the large diameter propeller tips reach the speed of sound. Both jet and piston engines experience reduced performance at higher **altitudes** due to decreased air density, but generally jet engines perform better at higher altitudes than piston engines. The better Range is due to higher speed and fuel efficiency

### Multi Engine (categories: Single Engine and Multi Engine)
Multiple Engines generally offer better **Speed**, **Range** and **Climb** performance.

### TP mods (categories: Modification or not)
This feature most likely refer to **Thrust Performance modifications** on Turbo Prop Engines (referred to as propjet in the data set) and is relevant only for the category propjet in the "Engine Type"-feature.  

### THR

<br>
<img src="image_readme/equations/eq_thr.png" alt="Equation for" style="width: 40%;"/>
<br>

### SHP

<br>
<img src="image_readme/equations/eq_shp_tn.png" alt="Equation for" style="width: 30%;"/>
<br>

<br>
<img src="image_readme/equations/eq_shp_p.png" alt="Equation for" style="width: 30%;"/>
<br>

The SHP could also be calculated by a similar formula using the the engine speed in RPM instead of the velocity of the aircraft.

### Length
This feature is of little value from a design/performance perspective albeit it could be used for corelation. The part of the length between the wings and tail planes quarter chords would be of a greater interest since it dictates static and dynamic stability.

### Height
This feature is of an even smaller value than Length.

### Wing Span
Wing Span is the one single dimensional feature of real value in the dataset however even here the Wing Area would be an even more useful feature to have. Wing Span does not directly relate to Lift (via the classic Lift equation) however since the wingspan is quadratically proportional to wing area (assuming constant aspect ratio/mean chord) a correlation with Wing Span should be seen whenever there is a correlation with Wing Area.

<br>
<img src="image_readme/equations/eq_lift.png" alt="Equation for" style="width: 70%;"/>
<br>

### FW (Fuel Weight)
Fuel weight (together with "AUW") naturally have strong correlation with **Range** Since the more fuel you carry in relationship to the weight of the airplane, the further you can fly (please see the equation for Range).

Note also that the FW can be used in the Range Equation.

### MEW (Empty weight, a.k.a Manufacturer's Empty Weight )
The Empty weight would be interesting to plot against **Year of first flight** and **Aircraft Structure** (see Outlook-chapter) to see if, with new modern material and buildung techniques" the airplanes have become lighter. For this such a study it is important to use MEW rather than AUW.

<br>
<img src="image_readme/equations/eq_mew.png" alt="Equation for" style="width: 50%;"/>
<br>

### AUW (Gross weight, a.k.a All-Up Weight)
The All-up Weight have a strong correlation with **Wing area** (as do MEW of course however AUW is the more appropriate feature here) since the Lifting force that the wing produces need to counteract the weight and Wing Area is part of the lift equation (see Outlook-chapter) but also Wing Span (albeit Aspect ratios vary)

Note also that the AUW can be used in the Range Equation.

<br>
<img src="image_readme/equations/eq_auw.png" alt="Equation for SHP " style="width: 50%;"/>
<br>

### Vmax (Max speed)
Max Speed should have a strong correlation to both Propulsion type and Multi Engine (and probably TP mods).

<br>
<img src="image_readme/equations/eq_v_max.png" alt="Equation for SHP " style="width: 50%;"/>
<br>

### Vcruise (Cruise speed)
Cruise Speed should have a strong correlation to both Propulsion type and Multi Engine (and probably TP mods).

<br>
<img src="image_readme/equations/eq_v_cruise.png" alt="Equation for SHP" style="width: 45%;"/>
<br>

### Vstall (Stall speed)
Stall speed should have a strong correlation to AUW and the relationship with Wing Span (and even more Wing Area) and AUW
<br>
<img src="image_readme/equations/eq_v_stall.png" alt="Equation for SHP" style="width: 35%;"/>
<br>

### Hmax (Max altitude)
**Velocity** is trongly correlated and Albeit not explicit in the below equation Hmax is strongly related to **ROC** since Hmax has been reached when the ROC reaches zero. Wing Span (more than Wing area) should also have a strong correlation. FW and AUW 

<br>
<img src="image_readme/equations/eq_h_max.png" alt="Equation for SHP" style="width: 40%;"/>
<br>

### Hmax (One) (Max altitude with only one Engine)
See Hmax.

### ROC (Rate of Climb)
THR, Vmax and AUW
<br>
<img src="image_readme/equations/eq_roc.png" alt="Equation for SHP" style="width: 45%;"/>
<br>

### ROC (One) (Rate of Climb with only one Engine)
See ROC.

### VLo (Climb speed during normal take-off for a 50 ft obstacle)
AUW and Span, indirectly, via Wing Area since Span and wing Area is somewhat related to each other.
<br>
<img src="image_readme/equations/eq_v_lo.png" alt="Equation for SHP"  style="width: 35%;/">
<br>

### SLo (Takeoff ground run)
The takeoff ground run has a THR and AUW
<br>
<img src="image_readme/equations/eq_s_lo.png" alt="Equation for SHP" style="width: 40%;/">
<br>

### Vl (Landing speed during normal landing for a 50 ft obstacle)
The Vl has a strong collelation to Vstall as well as the FW and AUW.
<br>
<img src="image_readme/equations/eq_vl.png" alt="Equation for SHP" style="width: 50%;/">
<br>

### Sl (Takeoff ground run)
Sl will only weakly correlate to the data set features. 
<br>
<img src="image_readme/equations/eq_sl.png" alt="Equation for SHP" style="width: 50%;/">
<br>

### Range
The classic Range equation (The Breguet Range equation) shows the direct relationship on the relationship between the fuel and also, indirectly, Wing Span via Lift (see the lift equation under Wing Span).

Please note that the AUW can be used as the initial weight in the Range Equation and that AUW - FW can be used as the final weight (After the fuel is consumed).

<br>
<img src="image_readme/equations/eq_range.png" alt="Equation for SHP" style="width: 40%;/">
<br>
<br>
