#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import streamlit as st
import streamlit as st
import pickle
import numpy as np
model = pickle.load(open('model.pkl','rb'))
st.title('Vehicle CO2 Emissions Project')
st.image('./image1.jpg')
st.markdown('Welcome! You are currently using an application that will predict the range of CO2 emissions (g/km) based on the technical characteristics of your vehicle.')
st.markdown('Once you have input numbers for each variable, please click the 'Calculate' button in order to see the estimated emissions based on different categories.')
st.markdown('<h2 style="color: blue; text-align: center;">Technical Charateristics</h2>', unsafe_allow_html=True)
Mass=st.number_input('Mass (kg)',min_value=0, max_value=None)
Axel_width=st.number_input('Axel Width (mm)',min_value=0, max_value=None)
Engine_capacity=st.number_input('Engine Capacity (cm3)',min_value=0, max_value=None)
Engine_power=st.number_input('Engine Power (kw)',min_value=0, max_value=None)


def predict_CO2(Mass,Axel_width, Engine_capacity, Engine_power):
    input=np.array([[Mass,Axel_width, Engine_capacity, Engine_power]]).astype(np.float64)
    prediction=model.predict(input)
    return round(float(prediction))



safe_html ="""  
<div style="background-color:#80ff80; padding:10px >
<h2 style="color:white;text-align:center;">CO2 is in range of 0-103 g/km</h2>
</div>
 """
warn_html ="""  
<div style="background-color:#FFFF8F; padding:10px >
<h2 style="color:white;text-align:center;">CO2 is in range of 104-113 g/km</h2>
</div>
 """
danger_html ="""  
<div style="background-color:#F4BB44; padding:10px >
<h2 style="color:white;text-align:center;">CO2 is in range of 114-123 g/km</h2>
</div>
 """
horrible_html ="""  
<div style="background-color:#EE4B2B; padding:10px >
<h2 style="color:white;text-align:center;">CO2 is in range of 124-516 g/km</h2>
</div>
 """

if st.button("Calculate"):
    
    output = predict_CO2(Mass,Axel_width, Engine_capacity, Engine_power)

    if output == 0:
        st.markdown(safe_html,unsafe_allow_html=True)
    elif output == 1:
        st.markdown(warn_html,unsafe_allow_html=True)
    elif output == 2:
        st.markdown(danger_html,unsafe_allow_html=True)
    elif output == 3:
        st.markdown(horrible_html,unsafe_allow_html=True)
    table = {'Category':[0,1,2,3], 'Data Range':['0-25%','25-50%','50%-75%', '>75%'], 'Emissions Range (g/km)':['0-103','104-113','114-123','>124']}
    table_df = pd.DataFrame(table)
    selection = table_df[{'Category','Data Range','Emissions Range (g/km)'}]
    selection.set_index('Category', inplace=True)
    st.table(selection)