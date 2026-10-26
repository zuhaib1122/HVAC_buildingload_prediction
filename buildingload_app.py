import datetime
import streamlit as st
import pandas as pd
import joblib
# set the page layout
st.set_page_config(page_title='BlP')
@st.cache_data
def load_dataset(path):
    df = pd.read_csv(path)
    return df
@st.cache_resource
def load_model():
    return joblib.load("HVAC_BuildingLoad_Prediction.pkl")

data = load_dataset("HVAC Energy Data.csv")
model = load_model()

st.markdown("<h1 style= 'text-align: center';>HVAC Building Load Prediction</h1>", unsafe_allow_html=True)
st.write("Enter live operational conditions below to forecast the building load using our trained model.")
col_left, col_center, col_right = st.columns(3)
with col_center:
    with st.expander("Dataset Overview"):
        st.dataframe(data.head(10), use_container_width=True, hide_index=True)
    st.caption("Expand the 'Dataset Overview' button to examine the first few rows of dataset.")

col_input, col_output = st.columns([1, 1], wrap=False)
with col_input:
    col_1, col_2 = st.columns(2)
    with col_1:
        selected_date = st.date_input('Select Operational Date', value=datetime.date.today())
    with col_2:
        selected_time = st.time_input("Select Operational Time", value=datetime.datetime.now().time())
    # combine date time
    operational_datetime = datetime.datetime.combine(selected_date, selected_time)
    # extract day, hour , month
    hour = operational_datetime.hour
    day = operational_datetime.day
    month = operational_datetime.month
    outside_temp_c = st.number_input('Outside Temperature (C)', value=40, max_value=85, min_value=20)
    outside_temp_f = 9 * (outside_temp_c/5) + 32
    cwt = st.number_input("Cooling Water Temperature (C)", value=25, max_value=38, min_value=20)
    cwr = st.number_input('Chilled Water Rate (liter per second)', value=100, min_value=70, max_value=150)
    humidity = st.number_input("Humidity (%)", value=60, min_value=30, max_value=90)
    windspeed = st.number_input('Wind Speed (miles per hour)', value=5, min_value=0, max_value=25)
    button = st.button("Calculate predicted Building Load", use_container_width=True)

    if button:
        input_data = pd.DataFrame({'Date': [day], 'Month': [month], 'Hour': [hour], 'Cooling Water Temperature (C)': [cwt], 'Outside Temperature (F)': [outside_temp_f ], 'Humidity (%)': [humidity], 'Wind Speed (mph)': [windspeed], 'Chilled Water Rate (L/sec)': [cwr]})
        prediction = model.predict(input_data)[0]
    
        st.markdown("### *Predicted Result:*")
        st.metric("Estimated Building Load", f"{prediction:.2f} RT", delta= "With estimated Accuracy 96%")
with col_output:
    with st.container(border=True):
        st.markdown(
        """
### **Quick Guide**
This tool forecast the Building Load (cooling power required fora building) using machine learning model. )
**Variables include:**
- *Building Load (BL):* The total heat energy tht must be removed to maintain comfort.
- **Outside Temperature:** The impact of ambient weather heat on building structure.
- **Cooling Water Temperature:** The temperature of water returning from cooling tower.
- **Humidity:** The impact of ambient humidity.
- **Wind Speed:** The ambient air speed.
\n  This forecast helps technicians and engineers to optimize chiller performance, save energy and avoid system overloads in real time.
"""
    )