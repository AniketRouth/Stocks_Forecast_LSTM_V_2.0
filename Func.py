import streamlit as st
import plotly.graph_objs as go
import datetime as dt
import numpy as np
import pandas_datareader.data as web
import pandas as pd
from arch import arch_model
import yfinance as yf
yf.pdr_override()


Sp = yf.download(tickers = ['AAPL'], 
    data_source = 'stooq',start= '2007-03-09' , end = dt.datetime.today().strftime('%Y-%m-%d'))['Adj Close'].pct_change().dropna()

datelist = pd.bdate_range(dt.date.today(), periods= 10).tolist()
Dt = [ str(x).split()[0] +" | " + str(x.strftime('%A')) for x in datelist]
D = [ str(x).split()[0]  for x in datelist]


def Page(Select_box_option):
    start = '2007-03-09'
    end = dt.datetime.today().strftime('%Y-%m-%d')
    st.set_page_config(layout="wide")
    st.header("Share : Price & Volatality Prediction")
    st.sidebar.header("Companies")
    Index_Select_Box = st.sidebar.selectbox(label = "",
    options = Select_box_option.index,key="page",index = 3) 

    Movig_Avg_selectbox = st.sidebar.selectbox(label = "Moving Average",
    options = "7 14 21 30 50 100".split(),key="MA",index = 3)
    return Movig_Avg_selectbox,Index_Select_Box,start,end

def Layout(tit,bgcol):
    layout = go.Layout(
        title=tit,    
        plot_bgcolor=bgcol,
        hovermode="x",
        hoverdistance=100, # Distance to show hover label of data point
        spikedistance=1000,
        legend=dict(
            # Adjust click behavior
            itemclick="toggleothers",
            itemdoubleclick="toggle"
        ),
        xaxis=dict(
            title="Date",
            linecolor="#BCCCDC",
            showgrid=False,
            spikethickness=2,
            spikedash="dot",
            spikecolor="#999999",
            spikemode="across"
        ),
        yaxis=dict(
            title="Price",
            linecolor="#BCCCDC",
            showgrid=False,
            spikethickness=2,
            spikedash="dot",
            spikecolor="#888888",
            spikemode="across"
        ),height = 600,
        width = 600
    )
    return layout


def MAChart(Selected_Moving_Avg,dataframe,chart_title,chart_bg):

    MA = dataframe['Open'].rolling(int(Selected_Moving_Avg)).mean().dropna()

    Modified_dataframe_chart = go.Scatter(
                x=MA.index,
                y=MA.values,
                name = str(Selected_Moving_Avg)+" MA",
                marker_color = "#FFFFFF"
            )

    Original_dataframe_chart = go.Scatter(
                x=dataframe.index[int(Selected_Moving_Avg):],
                y=dataframe['Open'].values[int(Selected_Moving_Avg):],
                name = "Open",
                marker_color='#FF6700'
                
            )

    Figures = [Modified_dataframe_chart ,Original_dataframe_chart]
    Moving_Average_Plot = go.Figure(data=Figures, layout=Layout(chart_title,chart_bg))
    st.plotly_chart(Moving_Average_Plot,use_container_width=True,sharing="streamlit")

def Cluster(df,Garch_p_q,Stock):
    Lstm_Cluster = df.loc[Stock][1]
    Garch_Cluster = df.loc[Stock][2]
    pq =  Garch_p_q[Garch_Cluster]
    return Lstm_Cluster, Garch_Cluster, pq

def GARCH_Chart(Original,chart_title,chart_bg,p,q):
    st.header("Volatality of Daily Returns")

    
    Daily_Ret = (Original['Adj Close'].pct_change()).dropna()

   
    st.subheader("Forecast Using GARCH")
    #c1,c2 = st.columns(2)

    form = st.form("Choose p & q")
    g_p = form.selectbox('P',range(1,26),p,key='p')
    g_q = form.selectbox('Q',range(0,26),q,key='q')
    form.form_submit_button("Submit")

    
    
    am = arch_model(Daily_Ret.values, vol="Garch", p=g_p, q=g_q, dist="t")
    res = am.fit(disp='off')
    su = res.summary
    st.code(su)
    forecasts = res.forecast(reindex=True,horizon=10)

    Pred = pd.Series(np.sqrt(forecasts.variance.values[-1,:]), index=Dt,name='Volatality')
    GA = st.checkbox('Table',False,key = 'GT')
    if GA:
        st.table(Pred)
    else:
        chart = go.Scatter(
                x=D,
                y=Pred,
                name = "Forecasted Volatality",
                marker_color='#FFFFFF')
        Vol = [chart]
        vol = go.Figure(data=Vol, layout=Layout(chart_title,chart_bg))
        st.plotly_chart(vol,use_container_width=True,sharing="streamlit")
    save_for_VaR = st.button("Save")

    if save_for_VaR:
        return forecasts,res

def VaR_report(gm_forecast,gm_result,name):

    SP500 = arch_model(Sp.values, p = 8, q = 0,
    mean = 'constant', vol = 'GARCH', dist = 't',rescale = True)
    sp_result = SP500.fit(disp='off')
    pred = sp_result.forecast(horizon=10,reindex=False)




    mean_forecast = gm_forecast.mean.dropna()
    variance_forecast = gm_forecast.variance.dropna()
    std_res = gm_result.resid/gm_result.conditional_volatility
    s = pd.DataFrame(std_res)
    q_parametric = s.quantile(0.05)
    q_parametric = q_parametric.values
    VaR = mean_forecast.values + np.sqrt(variance_forecast.values) * q_parametric
    VaR = pd.DataFrame(VaR.reshape(-1,1),columns = ['VaR'],index = Sp.index[-10:] )

    resid_stock = gm_result.resid / gm_result.conditional_volatility
    resid_sp500 = sp_result.resid / sp_result.conditional_volatility
    

    correlation = np.corrcoef(resid_stock[-100:], resid_sp500[-100:])[0, 1]
    stock_beta = correlation * (gm_result.conditional_volatility[-100:] /
    sp_result.conditional_volatility[-100:])
    VaR['Market Risk'] = stock_beta[-10:]
    VaR.columns = ['VaR' , 'Market Risk']

    if stock_beta[-100:].mean()>1:
         beta =  np.round((stock_beta[-100:].mean() -1),3)
    elif stock_beta[-100:].mean()<1:
         beta = np.round((1 - stock_beta[-100:].mean()),3)
    else:
         beta = stock_beta[-100:].mean()  

    st.header('Value & Market Risk')
    st.table(VaR)
    st.subheader("10 Day 5% VaR for  {} : {}% ".format(name,np.round(VaR.mean().values[0]*100),3))
    st.subheader("Beta value is : {}".format(beta))
