'''python
pip install pyreader
import requests

url = "https://raw.githubusercontent.com/shokru/mlfactor.github.io/master/material/data_ml.RData"

with open("data_ml.RData", "wb") as f:
    f.write(requests.get(url).content)
import pyreadr
import pandas as pd

result = pyreadr.read_r("data_ml.RData")
data_raw = result['data_ml']
data_ml = data_raw.loc[
    (data_raw['date'] > '1999-12-31') &
    (data_raw['date'] < '2019-01-01')
]
    ]
    .sort_values(['date', 'stock_id'])
    .reset_index(drop=True)
)
data_ml.iloc[0:6, 0:6]
# 
import matplotlib.pyplot as plt
pd.Series(data_ml.groupby('date').size()).plot(figsize=(8,4)) # counting the number of assets for each date
plt.ylabel('nb_assets')    

features=list(data_ml.iloc[:,3:95].columns) # Keep the feature's column names (hard-coded, beware!)
features_short =["Div_Yld", "Eps", "Mkt_Cap_12M_Usd", "Mom_11M_Usd", 
                    "Ocf", "Pb", "Vol1Y_Usd"]

col_feat_Div_Yld=data_ml.columns.get_loc('Div_Yld') # finding the location of the column/feature Div_Yld
is_custom_date =data_ml['date']=='2000-02-29'       # creating a boolean index to filter on 
data_ml[is_custom_date].iloc[:,[col_feat_Div_Yld]].hist(bins=100) # using the hist 
plt.ylabel('count')

df_median=[]          #creating empty placeholder for temporary dataframe
df=[]                #creating empty placeholder for temporary dataframe
import numpy as np
df_median=data_ml[['date','R1M_Usd','R12M_Usd']].groupby(['date']).median() # computings medians for both labels at each date 
df_median.rename(columns={"R1M_Usd": "R1M_Usd_median", "R12M_Usd": "R12M_Usd_median"},inplace=True)
df = pd.merge(data_ml,df_median,how='left', on=['date'])             # join the dataframes
data_ml['R1M_Usd_C'] = np.where(df['R1M_Usd'] > df['R1M_Usd_median'], 1.0, 0.0) # Create the categorical labels
data_ml['R12M_Usd_C'] = np.where(df['R12M_Usd'] > df['R12M_Usd_median'], 1.0, 0.0) # Create the categorical labels
df_median=[]          #removing the temp dataframe to keep it light!
df=[]                 #removing the temp dataframe to keep it light!

separation_date = "2014-01-15"
idx_train=data_ml.index[(data_ml['date'] < separation_date)].tolist() 
idx_test=data_ml.index[(data_ml['date'] >= separation_date)].tolist() 

stock_ids_short=[]   # creating empty placeholder for temporary dataframe
stock_days=[]        # creating empty placeholder for temporary dataframe
stock_ids=data_ml['stock_id'].unique() # A list of all stock_ids
stock_days=data_ml[['date','stock_id']].groupby(['stock_id']).count().reset_index() # compute the number of data points per stock
stock_ids_short=stock_days.loc[stock_days['date'] == (stock_days['date'].max())] # Stocks with full data
stock_ids_short=stock_ids_short['stock_id'].unique() ### in order to get a list 
is_stock_ids_short=data_ml['stock_id'].isin(stock_ids_short)  
returns=data_ml[is_stock_ids_short].pivot(index='date',columns='stock_id',values='R1M_Usd') # compute returns in matrix format
'''
