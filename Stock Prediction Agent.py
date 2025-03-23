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

#Constraints
starting_balance = 10000 #USD
transaction_fee = 0.01 #1% applies to each Buy or Sell order.
simulation_days = pd.date_range("2025-03-24", "2025-03-28", freq='B')  # 5 trading days

test_days = pd.date_range("2025-03-17", "2025-03-21", freq='B')

#Real time variables
current_balance = starting_balance

#Feature Engineering(Fine)
def engineer_features(data):
    #Simple Moving Averages(SMA) for 5 and 20 days
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()

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

    print("RSI" + str(data['RSI']))

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

class TradingAgent:
    def __init__(self, capital=starting_balance):
        self.capital = capital
        self.shares = 0
        self.transaction_fee = transaction_fee
        self.history = []
        
    def trade(self, action, price, amount=0):
        if action == "Buy":
            total_cost = amount * price + (amount * price * self.transaction_fee)
            if self.capital >= total_cost:
                self.shares += amount
                self.capital -= total_cost
                self.history.append(f"Bought {amount} shares at ${price} each")
        elif action == "Sell":
            total_revenue = amount * price + (amount * price * self.transaction_fee)
            if self.shares >= amount:
                self.shares -= amount
                self.capital += total_revenue
                self.history.append(f"Sold {amount} shares at ${price} each")
        elif action == "Hold":
            self.history.append("Held position")
        
    def get_balance(self, current_price):
        return self.capital + self.shares * current_price

def run_simulation(data, model, agent):
    for current_day in simulation_days:
        day_data = data.loc[data.index.date == current_day.date()]
        if not day_data.empty:
            current_price = day_data['Close'].iloc[-1]
            X_today = day_data[['SMA_5', 'SMA_20', 'EMA_12', 'EMA_26', 'RSI']].iloc[-1].values.reshape(1, -1)
            prediction = model.predict(X_today)
            
            if prediction == 1:
                action = "Buy"
                amount_to_buy = agent.capital // current_price
                agent.trade(action, current_price, amount_to_buy)

            elif prediction == 0 and agent.shares > 0:
                action = "Sell"
                agent.trade(action, current_price, agent.shares)
                
            else:
                action = "Hold"
                agent.trade(action, current_price)

            print(f"Day: {current_day.date()} | Action: {action} | Balance: ${agent.get_balance(current_price):.2f}")

#Main
print("##########TSLA Stock prediction:ML Trading Agent for Tesla Stocks##########")

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
print("Time: ", formatted_datetime)

#Download Previous Tesla stock data
ticker = "TSLA"
start_date = "2025-01-01"
end_date = "2025-03-22"
tesla_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)  #auto_adjust=True is a form of data preparation because its not raw data and ignores corporate decisions

# print(tesla_data)
# tesla_data['Close'].plot(title="Tesla Stock Price")
# plt.show()

#Get current stock data for Tesla
stock = yf.Ticker(ticker)
current_price = round(stock.history(period="1d")['Close'].iloc[0], 2)
print(f"Current price of Tesla Inc Stock ({ticker}): {current_price} USD")

tesla_data = engineer_features(tesla_data)
model = train_model(tesla_data)

#Create agent and run simulation
# agent = TradingAgent(capital=starting_balance)
# run_simulation(tesla_data, model, agent)