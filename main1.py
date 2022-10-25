import streamlit as st
import pickle
import numpy as np 
import pandas as pd
#import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") 


model1=pickle.load(open('timeseries.pkl',"rb"))
daily_data_last_7=pd.read_csv('last_seven_days.csv', header=None)
data=pd.read_csv("forecasting.csv")



st.title("Forecast power consumption data")
st.sidebar.subheader("Select the number of days to Forecast from 2018-Aug-4")
days = st.sidebar.number_input('Days',min_value = 1,step = 1)

z=daily_data_last_7
z=np.array(z[0].tail(7))
for i in range(0,days):
    r=z[-7:]
    r=np.array([r])
    ranf_f=model1.predict(r)
    z=np.append(z,ranf_f)
    i=+1
future_pred=z[-days:]

    
    
future = pd.date_range(start='4/8/2018',periods=days,tz=None,freq = 'D')
future_df = pd.DataFrame(index=future)
future_df['Power Consumption'] = future_pred.tolist()

st.sidebar.write(f"Power consumption for {days}th day")
st.sidebar.write(future_df[-1:])
st.write(f"Power consumptionForecasted till {days}" )

st.write(future_df)
