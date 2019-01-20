import pandas as pd
import numpy as np
import tensorflow as tf


def main():
    ############################################################
    # Problem #1
    ############################################################
    team_list = [
        "Golden State",
        "49ers",
        "Giants",
        "Cavaliers",
        "Lakers",
        "Rams",
        "Yankees"
    ]
    print("Problem 1: ", team_list)

    team_series1 = pd.Series(team_list)
    print("Problem 1 - team_series 1:\n", team_series1)

    ############################################################
    # Problem #2
    ############################################################
    team_dict = {
        'Oakland-Basketball': 'Golden State',
        'SF-Football': '49ers',
        'SF-Baseball': 'Giants',
        'Cleveland-Basketball': 'Cavaliers',
        'LA-Basketball': 'Lakers',
        'LA-Football': 'Rams',
        'NY-Baseball': 'Yankees',
    }

    print("Problem 2: ", team_dict)
    team_series2 = pd.Series(team_dict)

    print("Problem 2 - team_series 2:\n", team_series2)
    ############################################################
    # Problem #3
    ############################################################
    team_series1.index = [
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven'
    ]
    print("Problem 3 - 5th for teams_series1 via loc: ",
          team_series1.loc['five'])
    print("Problem 3 - 5th for teams_series1 via iloc: ", team_series1.iloc[4])
    print("Problem 3 - 7th for teams_series1 via loc: ",
          team_series1.loc['seven'])
    print("Problem 3 - 7th for teams_series1 via iloc: ", team_series1.iloc[6])
    print("Problem 3 - 5th for teams_series2 via loc: ",
          team_series2.loc[[False, False, False, False, True, False, False]].iloc[0])
    print("Problem 3 - 5th for teams_series2 via iloc: ", team_series2.iloc[4])
    print("Problem 3 - 7th for teams_series2 via loc: ",
          team_series2.loc[[False, False, False, False, False, False, True]].iloc[0])
    print("Problem 3 - 7th for teams_series2 via iloc: ", team_series2.iloc[6])
    ############################################################
    # Problem #4
    ############################################################
    num = pd.Series(range(1, 101, 1))
    series_sum = 0
    for series_int in num:
        series_sum = series_sum + series_int

    print("Problem 4 - sum by loop: ", series_sum)
    print("Problem 4 - sum by np: ", np.sum(series_sum))
    ############################################################
    # Problem #5
    ############################################################
    print("Problem 5 - 5 added to series num: ", (num+5))
    ############################################################
    # Problem #6
    ############################################################
    idx = range(1, 10)
    media_data = {"TV": pd.Series([230.1, 44.5, 17.2, 151.5, 180.8, 8.7, 57.5, 120.2, 8.6],
                                  index=idx),
                  "Radio": pd.Series([37.8, 39.3, 45.9, 41.3, 10.8, 48.9, 32.8, 19.6, 2.1],
                                     index=idx),
                  "NewsPaper": pd.Series([69.2, 45.1, 69.3, 58.5, 58.4, 75, 23.5, 11.6, 1],
                                         index=idx),
                  "Sales": pd.Series([22.1, 10.4, 9.3, 18.5, 12.9, 7.2, 11.8, 13.2, 4.8],
                                     index=idx)}
    print("Problem 6 - dataframe:\n", pd.DataFrame(media_data))
    ############################################################
    # Problem #7
    ############################################################
    house_data = pd.read_csv("00 kc_house_data.csv")
    print("Problem 7 - Number of observations in dataframe is ",
          (house_data.shape[0])*house_data.shape[1])
    print("Problem 7 - Average house price is ", house_data['price'].mean())
    bm = house_data['price'].gt(500000)
    print("Problem 7 - Number of houses which are priced greater than $500,000 is ",
          house_data[bm].shape[0])
    ############################################################
    # Problem #8
    ############################################################
    print("Problem 8 - tensorflow version is ", tf.__version__)


if __name__ == "__main__":
    main()
