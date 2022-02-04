# %%
import pandas as pd

# %%
data = pd.read_csv("../datasets/raw/market_insights_overview_5p.csv")

data["date"] = pd.to_datetime(data["date"])
data

# %%
categories = pd.read_csv("../datasets/raw/solr_categories_2021_11_29.csv")
categories_name = categories[["title", "internal_doc_id"]]
categories_name
# %%
grouped_joined= pd.merge(data, categories, how="left", left_on=["cat_id"], right_on=["internal_doc_id"])
grouped_joined.rename(columns={"title": "cat_name"}, inplace=True)
grouped_joined.head()
# %%

grouped = grouped_joined.groupby(["cat_id", "date"], as_index=False).sum()
grouped = grouped.merge(categories_name, how="left", left_on=["cat_id"], right_on=["internal_doc_id"])
grouped

# %%
counts = grouped["date"].value_counts()
counts.hist()

# %%
pivot = grouped.pivot( index="date", columns=["title", "cat_id"])


# %%
pivot.loc[[pivot["cat_id"] == "30"]]
pivot.head()
