
import pandas as pd

data = pd.read_csv("./datasets/market_insights_overview_all.csv")
data["date"] = pd.to_datetime(data["date"])

print(data.head())
