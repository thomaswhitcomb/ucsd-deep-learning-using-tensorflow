import pandas as pd

team_list = ["Golden State","49ers","Giants","Cavaliers","Lakers","Rams","Yankees"]
team_dict = {"Oakland-Basketball" :"Golden State","SF-Football":"49ers","SF-Baseball":"Giants","Cleveland-Basketball":"Cavaliers","LA-Basketball":"Lakers","LA-Football":"Rams","NY-Baseball":"Yankees"}

team_series1 = pd.Series(team_list)
team_series2 = pd.Series(team_dict)
