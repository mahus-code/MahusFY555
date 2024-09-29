import pandas as pd
import numpy as np

df = pd.DataFrame({'X':[78,85,96,80,86], 'Y':[4,94,89,83,86], 'Z':[6,97,96,72,83], 'C':[16,96,65,70,76]})

ds = pd.Series([2,4,6,8,10])
print(ds)

df_1 = pd.DataFrame(['100', '200', 'python', '300.12', '400'])
print(df_1)

df_2 = pd.DataFrame(['10', '20', 'php', '30.12', '40'])
print(df_2)

df_combined = pd.concat([df_1,df_2], axis=1)
print(df_combined)

def rename_df(data_frame):
    rename_dict = {col: f'column{index + 1}' for index, col in enumerate(data_frame.columns)}
    renamed_df = data_frame.rename(columns = rename_dict)
    return renamed_df
   
print(rename_df(df))
