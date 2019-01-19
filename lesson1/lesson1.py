import pandas as pd
import numpy as np

############################################################
# Problem #1
############################################################
team_list = ["Golden State","49ers","Giants","Cavaliers","Lakers","Rams","Yankees"]

team_series1 = pd.Series(team_list)
team_series1.index = ['one','two','three','four','five','six','seven']

  
############################################################
# Problem #2
############################################################
team_dict = {"Oakland-Basketball" :"Golden State","SF-Football":"49ers","SF-Baseball":"Giants","Cleveland-Basketball":"Cavaliers","LA-Basketball":"Lakers","LA-Football":"Rams","NY-Baseball":"Yankees"}
team_series2 = pd.Series(team_dict)
team_series2.index = ["Oakland-Basketball","SF-Football","SF-Baseball","Cleveland-Basketball","LA-Basketball","LA-Football","NY-Baseball"]

############################################################
# Problem #3
############################################################
if team_series1.iloc[4] == team_series1.loc['five'] and team_series1.iloc[6] == team_series1.loc['seven']:
  print("Problem #3 series1 correct")
else:
  print("Problem #3 series1 is incorrect")

if team_series2.iloc[4] == team_series2.loc[[False,False,False,False,True,False,False]].iloc[0] and team_series2.iloc[6] == team_series2.loc[[False,False,False,False,False,False,True]].iloc[0]:
  print("Problem #3 series2 correct")
else:
  print("Problem #3 series2 is incorrect")

############################################################
# Problem #4
############################################################
num = pd.Series(range(1,101,1))
sum = 0
for x in num:
  sum = sum + x

if sum == 5050 and sum == np.sum(num):
  print("Problem #4 is correct")
else:
  print("Problem #4 is incorrect")
############################################################
# Problem #5
############################################################
if np.sum(num + 5) == 5550:
  print("Problem #5 is correct")
else:
  print("Problem #5 is not correct")
############################################################
# Problem #6
############################################################
idx = range(1,10)
d = {"TV":pd.Series([230.1,44.5,17.2,151.5,180.8,8.7,57.5,120.2,8.6],index=idx),
     "Radio":pd.Series([37.8,39.3,45.9,41.3,10.8,48.9,32.8,19.6,2.1],index=idx),
     "NewsPaper":pd.Series([69.2,45.1,69.3,58.5,58.4,75,23.5,11.6,1],index=idx),
     "Sales":pd.Series([22.1,10.4,9.3,18.5,12.9,7.2,11.8,13.2,4.8],index=idx)}
p6 = pd.DataFrame(d)
print(p6)
############################################################
# Problem #7
############################################################
df = pd.read_csv("00 kc_house_data.csv")
print("Number of observations in dataframe is ",(df.shape[0])*df.shape[1])
print("Average house price is ",df['price'].mean())
bm = df['price'].gt(500000)
print("Number of houses which are priced greater than $500,000 is ",df[bm].shape[0])
