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

def titanic2():
	train = pd.read_csv("input/train.csv").copy()
	test = pd.read_csv("input/test.csv").copy()

	X = train.drop(["Cabin", "Name", "Ticket", "Survived", "PassengerId"], axis=1)
	X["Sex"] = X["Sex"].replace(["male", "female"], [0, 1])
	X["Embarked"] = X["Embarked"].replace(["S", "C", "Q"], [0, 1, 2])
	X["Embarked"] = X["Embarked"].fillna(0)
	X["Age"] = X["Age"].fillna(X["Age"].mean())
	X["Child"] = float('NaN')
	X["Child"][X["Age"] < 18] = 1
	X["Child"][X["Age"] >= 18] = 0
	X["FamilySize"] = 1 + X["SibSp"] + X["Parch"]
	y = train["Survived"]

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

	clf = svm.SVC(kernel='linear').fit(X_train, y_train)
	accuracy = clf.score(X_test, y_test)
	print "Accuracy: %d" % accuracy

	Xt = test.drop(["Cabin", "Name", "Ticket", "PassengerId"], axis=1)
	Xt["Sex"] = Xt["Sex"].replace(["male", "female"], [0, 1])
	Xt["Embarked"] = Xt["Embarked"].replace(["S", "C", "Q"], [0, 1, 2])

	Xt["Embarked"] = Xt["Embarked"].fillna(0)
	Xt["Age"] = Xt["Age"].fillna(X["Age"].mean())
	Xt["Fare"] = Xt["Fare"].fillna(X["Fare"].mean())

	Xt["Child"] = float('NaN')
	Xt["Child"][Xt["Age"] < 18] = 1
	Xt["Child"][Xt["Age"] >= 18] = 0
	Xt["FamilySize"] = 1 + Xt["SibSp"] + Xt["Parch"]
	res_survived = clf.predict(Xt)

	res = pd.DataFrame({
		"PassengerId": test["PassengerId"],
		"Survived": res_survived})
	res.to_csv("prediction2.csv", index=False)

if __name__ == '__main__':
	# titanic1()
	titanic2()