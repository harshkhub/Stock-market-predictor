#Harsh Khubchandani
#Stock predictor using scikit-learn and data about the NASDAQ index from the yfinance lib

import yfinance as yf
import pandas as pd

# Fetching historical data for NASDAQ index from yfinance
nasdaq = yf.Ticker("^IXIC")
nasdaq = nasdaq.history(period="max") 

# Cleaning up the data by removing unwanted columns
del nasdaq["Dividends"]
del nasdaq["Stock Splits"]

# Creating a "Tomorrow" column by shifting the "Close" prices by one day
nasdaq["Tomorrow"] = nasdaq["Close"].shift(-1)

# Creating a binary "Target" column based on whether the price will go up or down
nasdaq["Target"] = (nasdaq["Tomorrow"] > nasdaq["Close"]).astype(int)


nasdaq = nasdaq.loc["1990-01-01":].copy()

from sklearn.ensemble import RandomForestClassifier

# Initializing a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = nasdaq.iloc[:-100]
test = nasdaq.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])

# Converting predictions into a Pandas Series
preds = pd.Series(preds, index = test.index)

precision = precision_score(test["Target"], preds)

combined = pd.concat([test["Target"], preds], axis = 1)

# Defining a function to make predictions and combine actual and predicted values
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined

# Defining a function to perform backtesting
def backTest(data, model, predictors, start = 2500, step = 250):
    total_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train, test, predictors, model)
        total_predictions.append(predictions)
    return pd.concat(total_predictions)

predictions = backTest(nasdaq, model, predictors)

predictions["Predictions"].value_counts()

precision_score(predictions["Target"], predictions["Predictions"])

predictions["Target"].value_counts() / predictions.shape[0]

# Defining time horizons for additional features 2 days, 1 week, 3 months, 1 year, 5 years
horizons = [2, 5, 60, 250, 1000]

# Initializing a list to store the names of additional predictors to improve precision score
inc_predictors = []

for i in horizons:
    changing_averages = nasdaq.rolling(i).mean() # Calculting rolling mean over time horizons

    ratio_col = f"Close_ratio_{i}"
    nasdaq[ratio_col] = nasdaq["Close"] / changing_averages["Close"] #New predictor: ratio of close and average close

    trend_col = f"Trend_{i}"
    nasdaq[trend_col] = nasdaq.shift(1).rolling(i).sum()["Target"] #New predictor: sum of targets over various horizons

    inc_predictors += [ratio_col, trend_col]

nasdaq = nasdaq.dropna() # Removing NaN values 

#Changing machine learning model parameters 
model = RandomForestClassifier(n_estimators= 200, min_samples_split=50, random_state=1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    preds[preds >= .6] = 1
    preds[preds < .6] = 0
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined

predictions = backTest(nasdaq, model, inc_predictors)

print(predictions["Predictions"].value_counts())

print(precision_score(predictions["Target"], predictions["Predictions"]))

print(nasdaq)











