#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'preston.zhu'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model

import pdb

def set_ch():
	from pylab import mpl
	mpl.rcParams['font.sans-serif'] = ['SimHei']
	mpl.rcParams['axes.unicode_minus'] = False

def features1():
	train = pd.read_csv("input/train.csv")

	set_ch()
	fig = plt.figure()
	fig.set(alpha=0.2)

	plt.subplot2grid((2, 3), (0, 0))
	train.Survived.value_counts().plot(kind='bar')
	plt.title(u"获救情况（1为获救）")
	plt.ylabel(u"人数")

	plt.subplot2grid((2, 3), (0, 1))
	train.Pclass.value_counts().plot(kind='bar')
	plt.ylabel(u"人数")
	plt.title(u"乘客等级分布")

	plt.subplot2grid((2, 3), (0, 2))
	plt.scatter(train.Survived, train.Age)
	plt.ylabel(u"年龄")
	plt.grid(b=True, which='major', axis='y')
	plt.title(u"按年龄看获救分布（1为获救）")

	plt.subplot2grid((2, 3), (1, 0), colspan=2)
	train.Age[train.Pclass == 1].plot(kind='kde')
	train.Age[train.Pclass == 2].plot(kind='kde')
	train.Age[train.Pclass == 3].plot(kind='kde')
	plt.xlabel(u"年龄")
	plt.ylabel(u"密度")
	plt.title(u"各等级的乘客年龄分布")
	plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')

	plt.subplot2grid((2, 3), (1, 2))
	train.Embarked.value_counts().plot(kind='bar')
	plt.title(u"各登船口岸上船人数")
	plt.ylabel(u"人数")

	plt.show()

def features2():
	train = pd.read_csv("input/train.csv")

	set_ch()

	Survived_0 = train.Pclass[train.Survived == 0].value_counts()
	Survived_1 = train.Pclass[train.Survived == 1].value_counts()
	df1 = pd.DataFrame({u"获救": Survived_1, u"未获救": Survived_0})
	df1.plot(kind='bar', stacked=True)
	plt.title(u"各乘客等级的获救情况")
	plt.xlabel(u"乘客等级")
	plt.ylabel(u"人数")

	Survived_m = train.Survived[train.Sex == 'male'].value_counts()
	Survived_f = train.Survived[train.Sex == 'female'].value_counts()
	df2 = pd.DataFrame({u'男性': Survived_m, u'女性': Survived_f})
	df2.plot(kind='bar', stacked=True)
	plt.title(u"按性别看获救情况")
	plt.xlabel(u"性别")
	plt.ylabel(u"人数")
	plt.show()

	fig = plt.figure()
	fig.set(alpha=0.65)
	plt.title(u"根据舱等级和性别的获救情况")

	ax1 = fig.add_subplot(141)
	# ax1 = plt.subplot(141)
	train.Survived[train.Sex == 'female'][train.Pclass != 3].value_counts().plot(kind='bar', label="female high class", color='#FA2479')
	ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
	plt.legend([u"女性/高级舱"], loc='best')

	ax2 = fig.add_subplot(142, sharey=ax1)
	# ax2 = plt.subplot(142)
	train.Survived[train.Sex == 'female'][train.Pclass == 3].value_counts().plot(kind='bar', label="female, low class", color='pink')
	ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
	plt.legend([u"女性/低级舱"], loc='best')

	ax3 = fig.add_subplot(143, sharey=ax1)
	# ax3 = plt.subplot(143)
	train.Survived[train.Sex == 'male'][train.Pclass != 3].value_counts().plot(kind='bar', label="male, high class", color='lightblue')
	ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
	plt.legend([u"男性/高级舱"], loc='best')

	ax4 = fig.add_subplot(144, sharey=ax1)
	# ax4 = plt.subplot(144)
	train.Survived[train.Sex == 'male'][train.Pclass == 3].value_counts().plot(kind='bar', label="male, low class", color='steelblue')
	ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
	plt.legend([u"男性/低级舱"], loc='best')

	# plt.subplots_adjust(hspace=0.25, wspace=0.35)
	plt.show()

	# fig = plt.figure()
	# fig.set(alpha=0.2)

	Survived_0 = train.Embarked[train.Survived == 0].value_counts()
	Survived_1 = train.Embarked[train.Survived == 1].value_counts()
	df = pd.DataFrame({u"获救": Survived_1, u"未获救": Survived_0})
	df.plot(kind='bar', stacked=True)
	plt.title(u"各登录港口乘客的获救情况")
	plt.xlabel(u"登录港口")
	plt.ylabel(u"人数")

	plt.show()

	# g = train.groupby(['SibSp', 'Survived'])
	# df = pd.DataFrame(g.count()['PassengerId'])
	# print df

	# g = train.groupby(['Parch', 'Survived'])
	# df = pd.DataFrame(g.count()['PassengerId'])
	# print df

	Survived_cabin = train.Survived[pd.notnull(train.Cabin)].value_counts()
	Survived_nocabin = train.Survived[pd.isnull(train.Cabin)].value_counts()
	df = pd.DataFrame({u'有': Survived_cabin, u'无': Survived_nocabin}).transpose()
	df.plot(kind='bar', stacked=True)
	plt.title(u"按Cabin有无看获救情况")
	plt.xlabel(u"Cabin有无")
	plt.ylabel(u"人数")
	plt.show()

### Using RandomForestClassifier impute missing property 'Age'
def set_missing_ages(df):
	age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
	known_age = age_df[age_df.Age.notnull()].as_matrix()
	unknown_age = age_df[age_df.Age.isnull()].as_matrix()

	y = known_age[:, 0]
	X = known_age[:, 1:]

	rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
	rfr.fit(X, y)
	predictedAges = rfr.predict(unknown_age[:, 1:])
	df.loc[(df.Age.isnull()), 'Age'] = predictedAges

	return df, rfr

def set_Cabin_type(df):
	df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
	df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
	return df


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

def titanic3():
	train = pd.read_csv("input/train.csv")

	train, rfr = set_missing_ages(train)
	train = set_Cabin_type(train)

	# print train.head(10)

	# get_dummies()将该列数据分成若干列，列名为prefix前缀+类别名，列数据则为该类出现的位置为1，未出现则为0
	dummies_Cabin = pd.get_dummies(train['Cabin'], prefix='Cabin')
	dummies_Embarked = pd.get_dummies(train['Embarked'], prefix='Embarked')
	dummies_Sex = pd.get_dummies(train['Sex'], prefix='Sex')
	dummies_Pclass = pd.get_dummies(train['Pclass'], prefix='Pclass')

	df = pd.concat([train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
	df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
	# print df.head(10)

	scaler = preprocessing.StandardScaler()
	age_scale_param = scaler.fit(df['Age'])
	df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
	fare_scale_param = scaler.fit(df['Fare'])
	df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
	# print df.head(10)

	train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
	train_np = train_df.as_matrix()

	y = train_np[:, 0]
	X = train_np[:, 1:]

	clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
	clf.fit(X, y)

	test = pd.read_csv("input/test.csv")
	test.loc[(test.Fare.isnull()), 'Fare'] = 0
	tmp_df = test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
	null_age = tmp_df[test.Age.isnull()].as_matrix()
	X = null_age[:, 1:]
	predictedAges = rfr.predict(X)
	test.loc[(test.Age.isnull()), 'Age'] = predictedAges

	test = set_Cabin_type(test)

	dummies_Cabin = pd.get_dummies(test['Cabin'], prefix='Cabin')
	dummies_Embarked = pd.get_dummies(test['Embarked'], prefix='Embarked')
	dummies_Sex = pd.get_dummies(test['Sex'], prefix='Sex')
	dummies_Pclass = pd.get_dummies(test['Pclass'], prefix='Pclass')

	df_test = pd.concat([test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
	df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
	# 使用与训练集相同的参数进行特征缩放
	df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
	df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)

	test_set = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
	predictions= clf.predict(test_set)
	result = pd.DataFrame({
		'PassengerId': test['PassengerId'].as_matrix(),
		'Survived': predictions.astype(np.int32)
	})
	result.to_csv("prediction3.csv", index=False)

if __name__ == '__main__':
	# titanic1()
	# titanic2()
	# features1()
	# features2()
	titanic3()