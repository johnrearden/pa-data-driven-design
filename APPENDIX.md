# APPENDIX 

## Business case for unfinished Multi Engine Classification Model
- We want an ML model to predict if an Airplane have a Multi Engine or single Engine on historical General Aviation Airplane data. The target variable is categorical and contains 2-classes. We consider a **classification model**. It is a supervised model, a 2-class, single-label, classification model output: 0 (no Multi Engine), 1 (yes Multi Engine).
- Our ideal outcome is to provide our client with a predictor tool that can assist in feasibility studies of new proposal.
- The model success metrics are
	- at least 80% Recall for Multi Engine, on train and test set 
	- The ML model is considered a failure if:
		- Precision for no Multi Engine is lower than 80% on train and test set. (We don't want wrong decisions on configuration, such as Single or Multi Engine to be passed on to the Preliminary design phase potentially resulting in extremely costly redesigns if the mistake  isn ot discovered until the Preliminary Design phase or, even worse, in the Detailed Design phase)
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
| Length            | Wing Span, Length, Height, Slo and Sl           | ft and in                                   |    m     |  
| Distance          | Range                        | N.m. (Nautical miles)                       |   km     |  
| Weight            | FW, AUW and MEW            | lb                                          | kg or N  |
| Velocity          | Vmax, Vcruise, Vstall,          | knot or Mach and in                         |   m/s    |   
| Vertical velocity | ROC, Vlo and Vl          | ft/min                                      |   m/s    |

## Domain specific comments on relationships between the features in the data set
Outlined below are the dependencies between the features in the data set (and features mentioned in the Outlook) relevant for making hypotheses. Other dataset features are encircled in red as they appear in the equations. Underlined features indicate that they are indirectly related to other features in the dataset however the selection of which features to underline is rather ambiguous.

### Engine Type (categories: Piston and propjet, jet)
Jet generally offers higher **speed** and **ceilings** as well as better **range**. Propjet generally falls somewhere between these two engine types. Piston powered propeller driven propulsion units meets an invisible "speed barrier" approaching 400 knots. One reasons for this "barrier" is because the large diameter propeller tips reach the speed of sound. Both jet and piston engines experience reduced performance at higher **altitudes** due to decreased air density, but generally jet engines perform better at higher altitudes than piston engines. The better Range is due to higher speed and fuel efficiency

### Multi Engine (categories: Single Engine and Multi Engine)
Multiple Engines generally offer better **Speed**, **Range** and **Climb** performance.

### TP mods (categories: Modification or not)
This feature most likely refer to **Thrust Performance modifications** on Turbo Prop Engines (referred to as propjet in the data set) and is relevant only for the category propjet in the "Engine Type"-feature.  

### THR

$$ THR = \frac{T \times N}{33000} $$

Where:
- **THR** is the **Thrust** produced by the engine, measured in pounds of force (lbf).
- **T** is the **Torque** produced by the engine, measured in foot-pounds (ft-lb).
- **N** is the **Engine Speed**, measured in revolutions per minute (RPM).
- **33000** is a constant used to convert the product of torque and engine speed into thrust units (pounds of force).

**Note:** The **THR** (Thrust) is the force that propels the aircraft forward, and is derived from the engine’s torque and rotational speed. This equation shows how engine output in terms of torque and RPM can be translated into the thrust necessary to move the aircraft.

### SHP

$$ SHP = \frac{T \times N}{5252} $$

Where:
- **SHP** is the **Shaft Horsepower**, the amount of mechanical power produced by the engine's shaft.
- **T** is the **Torque** produced by the engine, typically measured in foot-pounds (ft-lb).
- **N** is the **Engine Speed**, measured in revolutions per minute (RPM).
- **5252** is a constant used to convert the torque and RPM into horsepower units (based on the equation for power).

**Note:** The **SHP** can also be calculated similarly to the lift equation but using the engine speed (RPM) rather than the aircraft's velocity, as **SHP** is directly tied to the rotational speed of the engine's shaft.

The SHP could also be calculated by a similar formula using the the engine speed in RPM instead of the velocity of the aircraft.

### Length
This feature is of little value from a design/performance perspective albeit it could be used for corelation. The part of the length between the wings and tail planes quarter chords would be of a greater interest since it dictates static and dynamic stability.

### Height
This feature is of an even smaller value than Length.

### Wing Span
Wing Span is the one single dimensional feature of real value in the dataset however even here the Wing Area would be an even more useful feature to have. Wing Span does not directly relate to Lift (via the classic Lift equation) however since the wingspan is quadratically proportional to wing area (assuming constant aspect ratio/mean chord) a correlation with Wing Span should be seen whenever there is a correlation with Wing Area.

$$ L = \frac{1}{2} \rho V^2 C_L $$

Where:
- **L** is the **lift** force, the upward force generated by the wings.
- **ρ** (rho) is the **air density**, which is the mass per unit volume of air, typically measured in kg/m³.
- **V** is the **velocity** of the airflow relative to the aircraft, usually measured in meters per second (m/s).
- **C_L** is the **coefficient of lift**, a dimensionless number that varies with the shape of the wing and the angle of attack.

### FW (Fuel Weight)
Fuel weight (together with "AUW") naturally have strong correlation with **Range** Since the more fuel you carry in relationship to the weight of the airplane, the further you can fly (please see the equation for Range).

Note also that the FW can be used in the Range Equation.

$$ FW = \text{Total Fuel Capacity} - \text{Residual Fuel Weight} $$

Where:
- **FW** is the **Fuel Weight**, the total weight of the fuel on board the aircraft.
- **Total Fuel Capacity** is the maximum amount of fuel the aircraft can hold, typically given in units of volume (like gallons or liters) or weight (kilograms or pounds).
- **Residual Fuel Weight** is the weight of the fuel remaining after a portion has been consumed during flight, or the amount of fuel left when the aircraft reaches its destination or required fuel reserves.

**Note:** Fuel Weight (**FW**) is closely related to the **Aircraft's Range**, as more fuel allows for a longer flight distance. **FW** is also a critical component in the **Range Equation**.

### MEW (Empty weight, a.k.a Manufacturer's Empty Weight )
The Empty weight would be interesting to plot against **Year of first flight** and **Aircraft Structure** (see Outlook-chapter) to see if, with new modern material and building techniques" the airplanes have become lighter. For this such a study it is important to use MEW rather than AUW.

$$ MEW = \text{Basic Empty Weight} + \text{Optional Equipment Weight} $$

Where:
- **MEW** (Manufacturer's Empty Weight) is the weight of the aircraft as it is delivered by the manufacturer, including everything except for fuel, passengers, and cargo.
- **Basic Empty Weight** is the weight of the aircraft with all standard equipment and systems installed, including the airframe, engines, and standard avionics, but without optional equipment, fuel, or passengers.
- **Optional Equipment Weight** is the weight of any optional equipment that is installed in the aircraft, such as additional avionics, seats, or other customized features.

### AUW (Gross weight, a.k.a All-Up Weight)
The All-up Weight have a strong correlation with **Wing area** (as do MEW of course however AUW is the more appropriate feature here) since the Lifting force that the wing produces need to counteract the weight and Wing Area is part of the lift equation (see Outlook-chapter) but also Wing Span (albeit Aspect ratios vary)

Note also that the AUW can be used in the Range Equation.

$$ AUW = MEW + FW + Payload $$

Where:
- **AUW** is the **All-Up Weight** (Gross Weight), which is the total weight of the aircraft when it is fully loaded, including the manufacturer’s empty weight, fuel, and payload.
- **MEW** is the **Manufacturer's Empty Weight**, which is the weight of the aircraft with all standard equipment installed but without fuel, passengers, or cargo.
- **FW** is the **Fuel Weight**, the total weight of the fuel on board the aircraft.
- **Payload** refers to the weight of passengers, cargo, and any other items carried in the aircraft.

**Note:** The **AUW** (All-Up Weight) is crucial in aircraft design because it is the weight that the wings must support, and it has a strong correlation with **Wing Area**. The **Lifting force** produced by the wings must counteract the **AUW**, and the **Wing Area** is part of the **Lift Equation**. Since the **AUW** directly impacts the aircraft’s lift capabilities, it is also an important factor in the **Range Equation**.

### Vmax (Max speed)
Max Speed should have a strong correlation to both Propulsion type and Multi Engine (and probably TP mods).

$$ V_{max} = \sqrt{\frac{2 \times \text{Thrust}}{\rho \times S \times C_D}} $$

Where:
- **Vmax** is the **Maximum Speed** of the aircraft, the highest speed that can be achieved under given conditions.
- **Thrust** is the total **Thrust** produced by the engine(s), typically measured in pounds of force (lbf).
- **ρ** (rho) is the **air density**, typically measured in kg/m³, which affects the drag and lift generated at different speeds.
- **S** is the **Wing Area**, the surface area of the aircraft's wings, which impacts the lift and drag forces.
- **C_D** is the **Drag Coefficient**, a dimensionless number that represents the drag produced by the aircraft at a given speed and configuration.

**Note:** The **Vmax** is strongly influenced by the **Propulsion Type** (jet engines, turboprops, etc.) and the **Engine Configuration** (e.g., single-engine vs. multi-engine). These factors impact the total available thrust and therefore the maximum speed the aircraft can achieve. Additionally, modifications to the **Turboprop (TP)** system or engine power can further adjust the **Vmax**.

### Vcruise (Cruise speed)
Cruise Speed should have a strong correlation to both Propulsion type and Multi Engine (and probably TP mods).

$$ V_{cruise} = \sqrt{\frac{2 \times \text{Thrust}}{\rho \times S \times C_D}} $$

Where:
- **Vcruise** is the **Cruise Speed**, the optimal speed for efficient flight, typically used during the cruising phase of the flight.
- **Thrust** is the total **Thrust** produced by the engine(s), typically measured in pounds of force (lbf).
- **ρ** (rho) is the **air density**, which impacts drag and overall aircraft performance.
- **S** is the **Wing Area**, influencing both lift and drag forces.
- **C_D** is the **Drag Coefficient**, representing the drag forces on the aircraft during flight.

**Note:** The **Vcruise** is closely related to the aircraft's **Thrust-to-Weight Ratio**, **Drag** characteristics, and **Wing Area**. This speed is typically achieved when the aircraft is in level flight at a stable altitude, optimizing the balance between thrust and drag for fuel efficiency. It's influenced by the **Propulsion Type** (such as jet or turboprop) and engine configuration.

### Vstall (Stall speed)
Stall speed should have a strong correlation to AUW and the relationship with Wing Span (and even more Wing Area) and AUW

$$ V_{stall} = \sqrt{\frac{2 \times W}{\rho \times S \times C_L}} $$

Where:
- **Vstall** is the **Stall Speed**, the lowest speed at which the aircraft can maintain level flight without stalling.
- **W** is the **Weight** of the aircraft, typically in pounds or kilograms.
- **ρ** (rho) is the **air density**, which changes with altitude and weather conditions.
- **S** is the **Wing Area**, the surface area of the wings, which influences the amount of lift generated at a given speed.
- **C_L** is the **Coefficient of Lift**, a dimensionless number that depends on the angle of attack and the shape of the wing.

**Note:** The **Vstall** is inversely related to the **Wing Area** and the **Coefficient of Lift**. The larger the **Wing Area** or the higher the **C_L**, the lower the stall speed. This makes **Vstall** crucial in determining the aircraft’s low-speed performance and handling characteristics.

### Hmax (Max altitude)
**Velocity** is strongly correlated and Albeit not explicit in the below equation Hmax is strongly related to **ROC** since Hmax has been reached when the ROC reaches zero. Wing Span (more than Wing area) should also have a strong correlation. FW and AUW 

$$ H_{max} = \frac{T \times \ln\left(\frac{W}{W_f}\right)}{g \times \rho_0 \times S \times C_D} $$

Where:
- **Hmax** is the **Maximum Altitude**, the highest altitude the aircraft can reach while maintaining level flight.
- **T** is the **Thrust** produced by the engine(s), typically measured in pounds of force (lbf).
- **W** is the **Weight** of the aircraft, including fuel and payload.
- **W_f** is the **Fuel Weight** at the maximum altitude.
- **g** is the **acceleration due to gravity**, typically 9.81 m/s².
- **ρ₀** is the **sea-level air density** (at standard conditions), which decreases as altitude increases.
- **S** is the **Wing Area**, which influences the lift generated at different altitudes.
- **C_D** is the **Drag Coefficient**, which changes with altitude and the aircraft’s configuration.

**Note:** **Hmax** is strongly influenced by the **Thrust-to-Weight Ratio**, the decreasing **air density** with altitude, and the **Wing Area**. As the aircraft climbs, the **air density** decreases, and so does the engine's thrust, requiring a higher rate of fuel consumption to maintain flight. The **Hmax** is the point where the aircraft can no longer sustain level flight due to insufficient thrust relative to its weight and drag.

### Hmax (One) (Max altitude with only one Engine)
See Hmax.

### ROC (Rate of Climb)
THR, Vmax and AUW

$$ ROC = \frac{T \times (V_{max} - V_{stall})}{W} $$

Where:
- **ROC** is the **Rate of Climb**, the vertical speed of the aircraft, typically measured in feet per minute (ft/min) or meters per second (m/s).
- **T** is the **Thrust** produced by the engine(s), typically in pounds of force (lbf).
- **Vmax** is the **Maximum Speed**, the highest speed the aircraft can achieve.
- **Vstall** is the **Stall Speed**, the lowest speed at which the aircraft can maintain level flight without stalling.
- **W** is the **Weight** of the aircraft, typically in pounds or kilograms.

**Note:** The **Rate of Climb** is determined by the difference between the available thrust (which is impacted by speed) and the drag forces, divided by the weight of the aircraft. A larger difference between **Vmax** and **Vstall** generally results in a higher rate of climb. Other factors such as altitude, engine performance, and aircraft configuration will also influence the ROC.

### ROC (One) (Rate of Climb with only one Engine)
See ROC.

### VLo (Climb speed during normal take-off for a 50 ft obstacle)
AUW and Span, indirectly, via Wing Area since Span and wing Area is somewhat related to each other.

$$ V_{Lo} = \sqrt{\frac{2 \times W}{\rho \times S \times C_L}} $$

Where:
- **Vlo** is the **Climb Speed** during normal takeoff, measured in knots or miles per hour.
- **W** is the **Weight** of the aircraft, typically in pounds or kilograms.
- **ρ** (rho) is the **air density**, which decreases with altitude and affects both lift and drag.
- **S** is the **Wing Area**, which influences the lift generated by the aircraft.
- **C_L** is the **Coefficient of Lift**, a dimensionless number that depends on the angle of attack and wing shape.

**Note:** The **Vlo** is the minimum speed that ensures the aircraft has sufficient lift to clear a 50 ft obstacle (or other obstacles depending on regulation) shortly after takeoff. It is closely related to the aircraft's **Weight**, **Wing Area**, and **Lift Coefficient**. The **Vlo** must be sufficient to provide a safe climb rate after takeoff while maintaining adequate control authority.

This speed is crucial for ensuring the aircraft can meet **obstacle clearance requirements** during takeoff.

Let me know if you need more details or modifications!

### SLo (Takeoff ground run)
The takeoff ground run has a THR and AUW

$$ S_{Lo} = \frac{V_{Lo}^2}{2 \times a} $$

Where:
- **Slo** is the **Takeoff Ground Run**, the distance the aircraft covers on the runway during takeoff, typically measured in meters or feet.
- **VLo** is the **Climb Speed** during normal takeoff, measured in meters per second (m/s) or knots.
- **a** is the **Acceleration** during the takeoff roll, typically in meters per second squared (m/s²).

**Note:** The **Slo** is calculated using the relationship between the **takeoff speed (VLo)** and the **acceleration (a)** during the ground run. The acceleration depends on factors like engine thrust, drag, and aircraft weight, as well as runway conditions. The distance required for takeoff is typically influenced by the aircraft’s **thrust-to-weight ratio** and the available runway length.

This equation assumes a constant acceleration during the ground run and does not account for the variation in acceleration as the aircraft accelerates down the runway.

Let me know if you'd like further details or modifications!

### Vl (Landing speed during normal landing for a 50 ft obstacle)
The Vl has a strong correlation to Vstall as well as the FW and AUW.

$$ V_l = \sqrt{\frac{2 \times W}{\rho \times S \times C_L}} $$

Where:
- **Vl** is the **Landing Speed** during normal landing, measured in knots or miles per hour.
- **W** is the **Weight** of the aircraft, typically in pounds or kilograms.
- **ρ** (rho) is the **air density**, which decreases with altitude and affects both lift and drag.
- **S** is the **Wing Area**, which influences the lift generated by the aircraft.
- **C_L** is the **Coefficient of Lift**, a dimensionless number that depends on the angle of attack and wing shape.

**Note:** The **Vl** is the minimum speed required to safely perform a landing approach while clearing a 50 ft obstacle. The aircraft needs to maintain sufficient lift at this speed to avoid stalling and ensure it can clear obstacles during final approach. The **Vl** is influenced by the aircraft’s **weight**, **wing area**, and **coefficient of lift**.

This speed is crucial for ensuring safe **obstacle clearance** during the landing phase, especially when landing in areas with potential obstacles near the runway.

Let me know if you need further clarification or additional equations!

### Sl (Landing ground run)
Sl will only weakly correlate to the data set features. 

$$ S_{l} = \frac{V_{l}^2}{2 \times a} $$

Where:
- **Sl** is the **Landing Ground Run**, the distance the aircraft travels on the runway after landing, measured in meters or feet.
- **Vl** is the **Landing Speed** at the moment of touchdown, typically measured in meters per second (m/s) or knots.
- **a** is the **Deceleration** during the landing roll, typically in meters per second squared (m/s²).

**Note:** The **Sl** is calculated based on the **landing speed (Vl)** at touchdown and the **deceleration (a)** during the ground roll. The deceleration is affected by the braking capability of the aircraft, runway conditions, and the use of other deceleration aids such as reverse thrust or spoilers. The distance required for landing depends on the aircraft's **weight**, braking efficiency, and drag forces encountered after touchdown.

This equation assumes a constant deceleration after landing, though in real-world scenarios, the deceleration rate can vary as the aircraft slows down.

Let me know if you'd like any further clarification or additional details!

### Range
The classic Breguet Range equation that dates back to before 1920.) shows the direct relationship on the relationship between the fuel and also, indirectly, Wing Span via Lift (see the lift equation under Wing Span). Below eq apply to propeller driven airplanes.

Please note that the AUW can be used as the initial weight in the Range Equation and that AUW - FW can be used as the final weight (After the fuel is consumed).

**For Propeller driven airplanes**

$$ R = \frac{V_e \times f_e}{g \times c_T} \times \ln\left(\frac{m_i}{m_f}\right) $$

Where:
- **R** is the **Range** of the aircraft.
- **V_e** is the **Equivalent Airspeed**.
- **f_e** is the **Fuel Efficiency** term.
- **g** is the **Acceleration due to gravity** (9.81 m/s²).
- **c_T** is the **Thrust-Specific Fuel Consumption (TSFC)**, related to fuel consumption per unit of thrust.
- **m_i** is the **Initial Mass** (total weight at the start of the flight, including full fuel).
- **m_f** is the **Final Mass** (weight at the end of the flight after burning fuel).

**Note:** This equation applies to **propeller-driven airplanes**, where **TSFC** plays a significant role in fuel consumption. The logarithmic term reflects how the range increases with the fuel burn rate and weight reduction over time.

<br>

**For Jet powered airplanes**

$$ R = \frac{V_e}{g \times c_T} \times \ln\left(\frac{m_i}{m_f}\right) $$

Where:
- **R** is the **Range** of the aircraft.
- **V_e** is the **Equivalent Airspeed**.
- **g** is the **Acceleration due to gravity** (9.81 m/s²).
- **c_T** is the **Thrust-Specific Fuel Consumption (TSFC)**, which measures the fuel consumption per unit of thrust.
- **m_i** is the **Initial Mass** (total weight at the start of the flight).
- **m_f** is the **Final Mass** (weight at the end of the flight after fuel burn).

**Note:** For **jet aircraft**, the **TSFC** is often lower than for propeller-driven aircraft, and the fuel efficiency term **f_e** is not included because jet engines have a different relationship between thrust and fuel consumption, typically driven by **engine speed** and **efficiency** over time. Jets rely more on aerodynamic efficiency and higher speeds at cruise altitude.
