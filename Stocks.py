import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import pandas_datareader.data as web
from Func import *
from Lstm_Forecast import *
import yfinance as yf
yf.pdr_override()

Stocks = pd.read_pickle('Cluster_of_Companies.pkl')
Garch_p_q = pd.read_pickle('Garch_p_q_Dict')

Share_Price = dict()
Predicted_Price  = dict()


Selected_Movig_Avg, Selected_Company, start, end = Page(Stocks)


if Selected_Company not in Share_Price.keys():
    Share_Price[Selected_Company] = yf.download(tickers = [Stocks.loc[Selected_Company][0]],start= start , end = end)

Lstm ,Garch, pq = Cluster(Stocks,Garch_p_q,Selected_Company)

if Selected_Company not in Predicted_Price.keys():
    Predicted_Price[Selected_Company] = Predict(Lstm,Share_Price[Selected_Company]['Open'])

MAChart(Selected_Movig_Avg, Share_Price[Selected_Company], Selected_Company,"#333333")


LSTM(Selected_Company, Predicted_Price[Selected_Company], Share_Price[Selected_Company],"#333333",Lstm)

forecast,model = GARCH_Chart(Share_Price[Selected_Company],Selected_Company,"#333333",pq[0],pq[1])

VaR_report(forecast,model,Selected_Company)
