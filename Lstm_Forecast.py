import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
import plotly.graph_objs as go
import datetime as dt
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from Func import Layout


Lstm_Models = [tf.keras.models.load_model('C0.h5'),
tf.keras.models.load_model('C1.h5'),
tf.keras.models.load_model('C2.h5')]

def Predict(model,data):
    Scaler_object = MinMaxScaler()
    model = Lstm_Models[model]
    t = Scaler_object.fit_transform(data.values.reshape(-1,1)).reshape(-1,1)
    input = []
    for iter in range(100, t.shape[0]):
        input.append(t[iter-100:iter, 0])
    input = np.array(input)
    input = np.reshape(input, (input.shape[0], input.shape[1], 1))
    predicted = Scaler_object.inverse_transform(model.predict(input).reshape(-1,1))
    return predicted

def Forecast(data,model):
    d = data[-(data.shape[0]):].reshape(-1,1)
    model = Lstm_Models[model]
    Min = MinMaxScaler(feature_range=(0,1))
    fore = Min.fit_transform(d)
    fore = fore.reshape(-1,d.shape[0],1)
    q = Min.inverse_transform(model.predict(fore)).flatten()
    return q

def LSTM(name,Prediction,Original,bgcol,model):

    Modified_dataframe_chart = go.Scatter(
                        x=Original.index[100:],
                        y=Prediction.reshape(-1),
                        name = "Predicted Open Price",
                        marker_color = '#FFFFFF'
                    )

    Original_dataframe_chart = go.Scatter(
                        x=Original.index[100:],
                        y=Original['Open'].values[100:],
                        name = "Daily Open Price",
                        marker_color='#996699'
                        
                    )
        

    Figures = [Modified_dataframe_chart ,Original_dataframe_chart]
    Lstm_plot = go.Figure(data=Figures,layout=Layout(name,bgcol))
    
    st.plotly_chart(Lstm_plot,use_container_width=True,sharing="streamlit")
    
    T=Original['Open'].values[-100:]

    for c in range(10):
        l = Forecast(T[-100:],model)
        T = list(T)
        T.append(l[0])
        T =  np.array(T)
    
    Forecasted = T[-10:]
    datelist = pd.bdate_range(dt.date.today(), periods=10).tolist()
    Dt = [ str(x).split()[0] +" | " + str(x.strftime('%A')) for x in datelist]
    Table = pd.Series(Forecasted,Dt,name = "Open")
    st.subheader("{} Days Forecast".format(len(Forecasted)))

    Forecast_Chart = go.Scatter(
                        x=datelist,
                        y=Forecasted,
                        name = name,
                        marker_color='#996699'
                        )
    Fig = [Forecast_Chart]
    Forecast_plot = go.Figure(data=Fig,layout=Layout(name,bgcol))
    TA = st.checkbox('Table',False,key='LT')
    if TA:
        st.table(Table)
    else:
        st.plotly_chart(Forecast_plot,use_container_width=True,sharing="streamlit")

    

