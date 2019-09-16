# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 00:22:03 2019

@author: SP Srivastava
"""

import pandas
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

games = pandas.read_csv("games.csv")
print(games.columns)
print(games.shape)

plt.hist(games["average_rating"])

plt.show()
print(games[games["average_rating"] == 0].iloc[0])
print(games[games["average_rating"] > 0].iloc[0])

games = games[games["users_rated"] > 0]
games = games.dropna(axis=0)

plt.hist(games["average_rating"])
plt.show()
corrmat = games.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()
columns = games.columns.tolist()
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]

target = "average_rating"
from sklearn.cross_validation import train_test_split

train = games.sample(frac=0.8, random_state=1)
test = games.loc[~games.index.isin(train.index)]
print(train.shape)
print(test.shape)
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(train[columns], train[target])

from sklearn.metrics import mean_squared_error

predictions = model.predict(test[columns])

mean_squared_error(predictions, test[target])
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=1)
model.fit(train[columns], train[target])
predictions = model.predict(test[columns])
mean_squared_error(predictions, test[target])