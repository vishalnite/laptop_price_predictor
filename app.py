# To run: streamlit run app.py 

import streamlit as st
import pickle
import numpy as np
import math

# Import the model
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

# Setting tab title
st.set_page_config(page_title="Laptop Price Predictor")

# Setting application title
st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop', value = 1.5)

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen Size
screen_size = st.number_input('Screen Size', value = 14.0)

# Resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', 
'1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440',
'2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# Hard Disk
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

# Solid State Drive
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# Operating System
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # query
    
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    query = query.reshape(1, 12)
    predicted_price = math.ceil(np.exp(pipe.predict(query)[0]))
    st.subheader(f"The predicted price is Rs. {predicted_price}")