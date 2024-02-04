
import yfinance as yf
import pandas as pd

nasdaq = yf.Ticker("^IXIC")

nasdaq = nasdaq.history(period="max")


del nasdaq["Dividends"]
del nasdaq["Stock Splits"]

nasdaq["Tomorrow"] = nasdaq["Close"].shift(-1)


nasdaq["Target"] = (nasdaq["Tomorrow"] > nasdaq["Close"]).astype(int)


nasdaq = nasdaq.loc["1990-01-01":].copy()

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = nasdaq.iloc[:-100]
test = nasdaq.iloc[-100:]

predictors = ["Close", "Volume", "Open", "High", "Low"]
model.fit(train[predictors], train["Target"])

from sklearn.metrics import precision_score

preds = model.predict(test[predictors])

preds = pd.Series(preds, index = test.index)

precision = precision_score(test["Target"], preds)

combined = pd.concat([test["Target"], preds], axis = 1)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index = test.index, name = "Predictions")
    combined = pd.concat([test["Target"], preds], axis = 1)
    return combined

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

horizons = [2, 5, 60, 250, 1000]

inc_predictors = []

for i in horizons:
    changing_averages = nasdaq.rolling(i).mean()

    ratio_col = f"Close_ratio_{i}"
    nasdaq[ratio_col] = nasdaq["Close"] / changing_averages["Close"]

    trend_col = f"Trend_{i}"
    nasdaq[trend_col] = nasdaq.shift(1).rolling(i).sum()["Target"]

    inc_predictors += [ratio_col, trend_col]

nasdaq = nasdaq.dropna()

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











