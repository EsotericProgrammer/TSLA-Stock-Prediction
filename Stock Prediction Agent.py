import random
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf
import schedule
import time
import matplotlib.pyplot as plt

#Goal: Maximize Profits

#Constraints
starting_balance = 10000 #USD
transaction_fee = 0.01  # 1% applies to each Buy or Sell order.

#Real time variables
current_balance = starting_balance
num_of_stocks_owned = 0

#Historical variables
total_stocks_bought = 0
total_stocks_sold = 0

# #Feature Engineering

# #Simple Moving Average(SMA) for 5 days
# #Data Smoothing
# tesla_data['SMA_5'] = tesla_data['Close'].rolling(window=5).mean()

# #Relative Strength Index(RSI) for 14 days
# delta = tesla_data['Close'].diff()
# gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
# loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
# rs = gain / loss
# tesla_data['RSI'] = 100 - (100 / (1 + rs))

# #Fill missing values(NaNs) with the previous value or mean
# tesla_data.bfill(inplace=True)

# print(tesla_data)


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
end_date = "2025-03-15" #Change this to be dynamic??
tesla_data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)  #auto_adjust=True is a form of data preparation because its not raw data and ignores corporate decisions
# print(tesla_data)

tesla_data['Close'].plot(title="Tesla Stock Price")
plt.show()

#Get stock data for Tesla
# stock = yf.Ticker(ticker)
# current_price = round(stock.history(period="1d")['Close'][0], 2)
# print(f"Current price of Tesla Inc Stock ({ticker}): {current_price} USD")

# print("Agent's Recommendation:")
# decision()
# #input("Press Enter to exit...")


# # Define the task to run every second
# def job():
#     print("Executing the event every second!")

# # Schedule the job to run every second
# schedule.every(1).seconds.do(job)

# while True:
#     schedule.run_pending()  # Run pending scheduled jobs
#     time.sleep(1)  # Sleep for 1 second before checking again