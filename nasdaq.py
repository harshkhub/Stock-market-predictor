
import yfinance as yf
import matplotlib.pyplot as plt

nasdaq = yf.Ticker("^IXIC")

nasdaq = nasdaq.history(period="max")

print(nasdaq)

nasdaq.plot.line(y="Close", use_index = True)

del nasdaq["Dividends"]
del nasdaq["Stock Splits"]

plt.title("NASDAQ Composite Index Closing Prices")
plt.xlabel("Date")
plt.ylabel("Closing Price")
plt.show()
