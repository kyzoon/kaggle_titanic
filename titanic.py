#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'preston.zhu'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer

import pdb

def titanic1():
	train = pd.read_csv("input/train.csv")
	test = pd.read_csv("input/test.csv")

	X = train.drop(["Cabin", "Name", "Ticket", "Survived"], axis=1)
	X["Sex"] = X["Sex"].replace(["male", "female"], [0, 1])
	X["Embarked"] = X["Embarked"].replace(["S", "C", "Q"], [0, 1, 2])
	imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
	imp.fit(X)
	X = imp.transform(X)
	y = train["Survived"]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	clf = svm.SVC(kernel='linear').fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	print "Accuracy: %d" % accuracy

	X = test.drop(["Cabin", "Name", "Ticket"], axis=1)
	X["Sex"] = X["Sex"].replace(["male", "female"], [0, 1])
	X["Embarked"] = X["Embarked"].replace(["S", "C", "Q"], [0, 1, 2])
	imp = Imputer(missing_values="NaN", strategy="mean", axis=0)
	imp.fit(X)
	X = imp.transform(X)
	res_survived = clf.predict(X)

	res = pd.DataFrame({
		"PassengerId": test["PassengerId"],
		"Survived": res_survived})
	res.to_csv("prediction.csv", index=False)

if __name__ == '__main__':
	titanic1()