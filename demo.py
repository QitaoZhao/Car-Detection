import streamlit as st
import time
import pandas as pd
import numpy as np
import pydeck as pdk

if "celsius" not in st.session_state:
    # set the initial default value of the slider widget
    st.session_state.celsius = 50.0

st.slider(
    "Temperature in Celsius",
    min_value=-100.0,
    max_value=100.0,
    key="celsius"
)

# This will get the value of the slider widget
st.write(st.session_state.celsius)

my_bar = st.progress(0)

with st.spinner('Wait for it...'):
    for percent_complete in range(100):
        time.sleep(0.05)
        my_bar.progress(percent_complete + 1)

my_bar.empty()
good = st.progress(0)

df = pd.DataFrame(
    np.random.randn(1000, 2) / [50, 50] + [36.37344, 120.68088],
    columns=['lat', 'lon'])

st.pydeck_chart(pdk.Deck(
     map_style='mapbox://styles/mapbox/light-v9',
     initial_view_state=pdk.ViewState(
         latitude=36.37344,
         longitude=120.68088,
         zoom=11,
         pitch=50,
     ),
     layers=[
         pdk.Layer(
            'HexagonLayer',
            data=df,
            get_position='[lon, lat]',
            radius=200,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
         ),
         pdk.Layer(
             'ScatterplotLayer',
             data=df,
             get_position='[lon, lat]',
             get_color='[200, 30, 0, 160]',
             get_radius=200,
         ),
     ],
 ))