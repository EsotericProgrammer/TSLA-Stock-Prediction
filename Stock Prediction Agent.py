import random
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import schedule
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Goal: Maximize Profits

#Constraints
starting_balance = 10000 #USD
transaction_fee = 0.01  # 1% applies to each Buy or Sell order.
simulation_days = pd.date_range("2025-03-24", "2025-03-28", freq='B')  # 5 trading days

#Real time variables
current_balance = starting_balance
num_of_stocks_owned = 0

#Historical variables
total_stocks_bought = 0
total_stocks_sold = 0

#Feature Engineering
def engineer_features(data):
    #Simple Moving Averages(SMA) for 5 and 20 days
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    # print("SMA5" + str(data['SMA_5']))
    # print("SMA20" + str(data['SMA_20']))

    #Exponential Moving Averages(EMA)
    data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()

    #Relative Strength Index(RSI)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    #Drop missing values due to rolling calculations
    data = data.dropna()

    return data

#Model Training
def train_model(data):
    #Predict whether the price will go up (1) or down (0)
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    #Select features and target
    features = ['SMA_5', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI']
    X = data[features]
    y = data['Target']
    
    #Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    #Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    #Test the model
    y_pred = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
    
    return model

def decision():
    print("WIP")

def sell(amount):
    print("Sell " + str(amount) + " Stocks")

def buy(amount):
    print("Buy " + str(amount) + " Stocks")

def hold():
    print("Hold")


#Main
print("##########TSLA Stock prediction:ML Trading Agent for Tesla Stocks##########")

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
print("Time: ", formatted_datetime)

#Download Previous Tesla stock data
ticker = "TSLA"
start_date = "2025-01-01"
end_date = "2025-03-20" #Change this to be dynamic??
tesla_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)  #auto_adjust=True is a form of data preparation because its not raw data and ignores corporate decisions

# print(tesla_data)
# tesla_data['Close'].plot(title="Tesla Stock Price")
# plt.show()

#Get current stock data for Tesla
stock = yf.Ticker(ticker)
current_price = round(stock.history(period="1d")['Close'].iloc[0], 2)
print(f"Current price of Tesla Inc Stock ({ticker}): {current_price} USD")
print("Agent's Recommendation:")
# decision()

tesla_data = engineer_features(tesla_data)
model = train_model(tesla_data)