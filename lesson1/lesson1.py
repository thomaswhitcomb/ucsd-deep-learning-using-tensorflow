import pandas as pd

team_list = ["Golden State","49ers","Giants","Cavaliers","Lakers","Rams","Yankees"]

team_series1 = pd.Series(team_list)
team_series1.index = ['one','two','three','four','five','six','seven']

if team_series1.iloc[4] == team_series1.loc['five'] and team_series1.iloc[6] == team_series1.loc['seven']:
  print("Problem #3 series1 correct")
else:
  print("Problem #3 series1 is incorrect")

  
team_dict = {"Oakland-Basketball" :"Golden State","SF-Football":"49ers","SF-Baseball":"Giants","Cleveland-Basketball":"Cavaliers","LA-Basketball":"Lakers","LA-Football":"Rams","NY-Baseball":"Yankees"}
team_series2 = pd.Series(team_dict)
team_series2.index = ["Oakland-Basketball","SF-Football","SF-Baseball","Cleveland-Basketball","LA-Basketball","LA-Football","NY-Baseball"]

print(team_series2)
if team_series2.iloc[4] == team_series2.loc[[False,False,False,False,True,False,False]].iloc[0] and team_series2.iloc[6] == team_series2.loc[[False,False,False,False,False,False,True]].iloc[0]:
  print("Problem #3 series2 correct")
else:
  print("Problem #3 series2 is incorrect")
