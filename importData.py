import pandas as pd

df = pd.read_csv("data_test_csv.csv", delimiter=";",usecols=["Data3"])

print(df)