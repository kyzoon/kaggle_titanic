#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'preston.zhu'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
# import sklearn.preprocessing as preprocessing
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import learning_curve, ShuffleSplit
from sklearn.ensemble import BaggingRegressor

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

	plt.subplot(121)
	Survived_0 = train.Pclass[train.Survived == 0].value_counts()
	Survived_1 = train.Pclass[train.Survived == 1].value_counts()
	df1 = pd.DataFrame({u"获救": Survived_1, u"未获救": Survived_0})
	df1.plot(ax=plt.gca(), kind='bar', stacked=True)
	plt.title(u"各乘客等级的获救情况")
	plt.xlabel(u"乘客等级")
	plt.ylabel(u"人数")

	plt.subplot(122)
	Survived_m = train.Survived[train.Sex == 'male'].value_counts()
	Survived_f = train.Survived[train.Sex == 'female'].value_counts()
	df2 = pd.DataFrame({u'男性': Survived_m, u'女性': Survived_f})
	df2.plot(ax=plt.gca(), kind='bar', stacked=True)
	plt.title(u"按性别看获救情况")
	plt.xlabel(u"性别")
	plt.ylabel(u"人数")
	plt.show()

	#######################################
	plt.title(u"根据舱等级和性别的获救情况")

	# ax1 = fig.add_subplot(141)
	ax1 = plt.subplot(141)
	fl = train.Survived[train.Sex == 'female'][train.Pclass != 3].value_counts()
	fl.plot.bar(title="female, high class", color='#FA2479')
	ax1.set_xticklabels([u"获救", u"未获救"], rotation=0)
	plt.legend([u"女性/高级舱"], loc='best')

	# ax2 = fig.add_subplot(142, sharey=ax1)
	ax2 = plt.subplot(142, sharey=ax1)
	fh = train.Survived[train.Sex == 'female'][train.Pclass == 3].value_counts()
	fh.plot.bar(ax=ax2, title="female, low class", color='pink')
	ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
	plt.legend([u"女性/低级舱"], loc='best')

	# ax3 = fig.add_subplot(143, sharey=ax1)
	ax3 = plt.subplot(143, sharey=ax1)
	ml = train.Survived[train.Sex == 'male'][train.Pclass != 3].value_counts()
	ml.plot.bar(ax=ax3, title="male, high class", color='lightblue')
	ax3.set_xticklabels([u"未获救", u"获救"], rotation=0)
	plt.legend([u"男性/高级舱"], loc='best')

	# ax4 = fig.add_subplot(144, sharey=ax1)
	ax4 = plt.subplot(144, sharey=ax1)
	mh = train.Survived[train.Sex == 'male'][train.Pclass == 3].value_counts()
	mh.plot.bar(ax=ax4, title="male, low class", color='steelblue')
	ax4.set_xticklabels([u"未获救", u"获救"], rotation=0)
	plt.legend([u"男性/低级舱"], loc='best')

	# plt.subplots_adjust(hspace=0.25, wspace=0.35)
	plt.show()

	plt.subplot(121)
	Survived_0 = train.Embarked[train.Survived == 0].value_counts()
	Survived_1 = train.Embarked[train.Survived == 1].value_counts()
	df = pd.DataFrame({u"获救": Survived_1, u"未获救": Survived_0})
	df.plot(ax=plt.gca(), kind='bar', stacked=True)
	plt.title(u"各登录港口乘客的获救情况")
	plt.xlabel(u"登录港口")
	plt.ylabel(u"人数")

	# g = train.groupby(['SibSp', 'Survived'])
	# df = pd.DataFrame(g.count()['PassengerId'])
	# print df

	# g = train.groupby(['Parch', 'Survived'])
	# df = pd.DataFrame(g.count()['PassengerId'])
	# print df

	plt.subplot(122)
	Survived_cabin = train.Survived[pd.notnull(train.Cabin)].value_counts()
	Survived_nocabin = train.Survived[pd.isnull(train.Cabin)].value_counts()
	df = pd.DataFrame({u'有': Survived_cabin, u'无': Survived_nocabin}).transpose()
	df.plot(ax=plt.gca(), kind='bar', stacked=True)
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

	# 对单独列进行缩放，数据为1维，所以使用如下方式进行处理
	mapper = DataFrameMapper([(['Age', 'Fare'], StandardScaler())])
	df['Age_scaled'] = mapper.fit_transform(df)[:, 0]
	df['Fare_scaled'] = mapper.fit_transform(df)[:, 1]
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
	# result = pd.DataFrame({
	# 	'PassengerId': test['PassengerId'].as_matrix(),
	# 	'Survived': predictions.astype(np.int32)
	# })
	# result.to_csv("prediction3.csv", index=False)

	# 查看特征权重情况
	print pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
						train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
	train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)
	train_scores_mean = np.mean(train_scores, axis=1)
	train_scores_std = np.std(train_scores, axis=1)
	test_scores_mean = np.mean(test_scores, axis=1)
	test_scores_std = np.std(test_scores, axis=1)

	if plot:
		set_ch()
		plt.figure()
		plt.title(title)
		if ylim is not None:
			plt.ylim(*ylim)
		plt.xlabel(u"训练样本数")
		plt.ylabel(u"得分")
		plt.gca().invert_yaxis()
		plt.grid()

		plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
						 alpha=0.1, color="b")
		plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
						 alpha=0.1, color="r")
		plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label=u"训练集上得分")
		plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label=u"交叉验证集上得分")
		plt.legend(loc='best')
		plt.draw()
		plt.show()
		plt.gca().invert_yaxis()

	midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
	diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
	return midpoint, diff

def cv_train():
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

	# 对单独列进行缩放，数据为1维，所以使用如下方式进行处理
	mapper = DataFrameMapper([(['Age', 'Fare'], StandardScaler())])
	df['Age_scaled'] = mapper.fit_transform(df)[:, 0]
	df['Fare_scaled'] = mapper.fit_transform(df)[:, 1]
	# print df.head(10)

	split_train, split_cv = train_test_split(df, test_size=0.3, random_state=0)
	train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

	train_np = train_df.as_matrix()
	y = train_np[:, 0]
	X = train_np[:, 1:]

	clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
	cross_val_score(clf, X, y, cv=5)
	clf.fit(X, y)

	cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
	X_cv = cv_df.as_matrix()[:, 1:]
	y_cv = cv_df.as_matrix()[:, 0]
	predictions = clf.predict(X_cv)

	origin_train = pd.read_csv("input/train.csv")
	bad_cases = origin_train.loc[origin_train['PassengerId'].isin(split_cv[predictions != y_cv]['PassengerId'].values)]
	# print bad_cases

	cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)
	plot_learning_curve(clf, u"学习曲线", X, y, cv=cv)

def modelensemble():
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

	# 对单独列进行缩放，数据为1维，所以使用如下方式进行处理
	mapper = DataFrameMapper([(['Age', 'Fare'], StandardScaler())])
	df['Age_scaled'] = mapper.fit_transform(df)[:, 0]
	df['Fare_scaled'] = mapper.fit_transform(df)[:, 1]
	# print df.head(10)

	split_train, split_cv = train_test_split(df, test_size=0.3, random_state=0)
	train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')

	train_np = train_df.as_matrix()
	y = train_np[:, 0]
	X = train_np[:, 1:]

	clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
	bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0,
									bootstrap=True, bootstrap_features=False, n_jobs=-1)
	bagging_clf.fit(X, y)

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
	df_test['Age_scaled'] = mapper.fit_transform(df_test)[:, 0]
	df_test['Fare_scaled'] = mapper.fit_transform(df_test)[:, 1]

	test_set = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
	predictions = bagging_clf.predict(test_set)
	result = pd.DataFrame({
		'PassengerId':test['PassengerId'].as_matrix(),
		'Survived':predictions.astype(np.int32)
	})
	result.to_csv("prediction4.csv", index=False)

def preparing():
	train = pd.read_csv("input/train.csv")

	# 补充缺失值
	# Embarked 补充众数'S'
	train.Embarked = train.Embarked.fillna('S')
	# Age使用随机森林进行补充
	train, rfr = set_missing_ages(train)

	# Child
	train['Child'] = float(0)
	train.loc[train.Age < 12, 'Child'] = 1
	# Mother
	train['Mother'] = float(0)
	train.loc[(train.Name.str.find('Mrs') != -1) & (train.Parch > 1), 'Mother'] = 1
	# FamilySize
	train['FamilySize'] = float(1) + train.Parch + train.SibSp
	# Title
	train['Title'] = float(0)
	train.loc[(train.Name.str.find('Mr') == -1) & (train.Name.str.find('Mrs') == -1) & (train.Name.str.find('Miss') == -1), 'Title'] = 1
	# CabinType
	train['CabinType'] = float('NaN')
	train.loc[train.Cabin.notnull(), 'CabinType'] = train.Cabin.str[0]
	train.loc[(train.Cabin.isnull()) & (train.Survived == 1), 'CabinType'] = 'C'
	train.loc[(train.Cabin.isnull()) & (train.Survived == 0), 'CabinType'] = 'T'
	# dummy Pclass and Sex
	dummies_Pclass = pd.get_dummies(train['Pclass'], prefix='Pclass')
	dummies_Sex = pd. get_dummies(train['Sex'], prefix='Sex')
	dummies_CabinType = pd.get_dummies(train['CabinType'], prefix='CabinType')

	train = pd.concat([train, dummies_Sex, dummies_Pclass, dummies_CabinType], axis=1)

	# 去掉Name, Ticket, Embarked
	train = train.drop(['Name', 'Ticket', 'Embarked', 'Cabin', 'CabinType', 'PassengerId', 'Sex', 'Pclass', 'SibSp', 'Parch'], axis=1)

	# Features Scaling
	mapper = DataFrameMapper([(['Age', 'Fare', 'FamilySize'], StandardScaler())])
	train['Age'] = mapper.fit_transform(train)[:, 0]
	train['Fare'] = mapper.fit_transform(train)[:, 1]
	train['FamilySize'] = mapper.fit_transform(train)[:, 2]

	train_part, cv_part = train_test_split(train, test_size=0.2, random_state=0)
	X = train_part.as_matrix()[:, 1:]
	y = train_part.as_matrix()[:, 0]

	X_cv = cv_part.as_matrix()[:, 1:]
	y_cv = cv_part.as_matrix()[:, 0]

	#################### test ###########################
	test = pd.read_csv("input/test.csv")

	# 补充缺失值
	# Fare 补充平均值
	test.loc[test.Fare.isnull(), 'Fare'] = test.Fare.mean()
	# Age使用随机森林进行补充
	test, rfr = set_missing_ages(test)

	# Child
	test['Child'] = float(0)
	test.loc[test.Age < 12, 'Child'] = 1
	# Mother
	test['Mother'] = float(0)
	test.loc[(test.Name.str.find('Mrs') != -1) & (test.Parch > 1), 'Mother'] = 1
	# FamilySize
	test['FamilySize'] = float(1) + test.Parch + test.SibSp
	# Title
	test['Title'] = float(0)
	test.loc[(test.Name.str.find('Mr') == -1) & (test.Name.str.find('Mrs') == -1) & (test.Name.str.find('Miss') == -1), 'Title'] = 1
	# CabinType
	test['CabinType'] = float('NaN')
	test.loc[test.Cabin.notnull(), 'CabinType'] = test.Cabin.str[0]
	test.loc[test.Cabin.isnull(), 'CabinType'] = 'A'
	# dummy Pclass and Sex
	dummies_Pclass = pd.get_dummies(test['Pclass'], prefix='Pclass')
	dummies_Sex = pd. get_dummies(test['Sex'], prefix='Sex')
	dummies_CabinType = pd.get_dummies(test['CabinType'], prefix='CabinType')

	test = pd.concat([test, dummies_Sex, dummies_Pclass, dummies_CabinType], axis=1)
	test['CabinType_T'] = float(0)

	# 去掉Name, Ticket, Embarked
	test = test.drop(['Name', 'Ticket', 'Embarked', 'Cabin', 'CabinType', 'Sex', 'Pclass', 'SibSp', 'Parch'], axis=1)

	# Features Scaling
	test['Age'] = mapper.fit_transform(test)[:, 0]
	test['Fare'] = mapper.fit_transform(test)[:, 1]
	test['FamilySize'] = mapper.fit_transform(test)[:, 2]

	return X, y, X_cv, y_cv, test.as_matrix()[:, 1:], test.as_matrix()[:, 0]

def titanic5():
	X, y, X_cv, y_cv, X_test, test_id = preparing()

	clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
	cross_val_score(clf, X, y, cv=5)
	clf.fit(X, y)

	cv = ShuffleSplit(n_splits=100, test_size=0.3, random_state=0)
	plot_learning_curve(clf, u"学习曲线", X, y, cv=cv)

	print clf.score(X_cv, y_cv)

	res_predictions = clf.predict(X_test)
	result = pd.DataFrame({'PassengerId': test_id.astype(np.int32), 'Survived': res_predictions.astype(np.int32)})
	result.to_csv("prediction5.csv", index=False)

if __name__ == '__main__':
	# titanic1()
	# titanic2()
	# features1()
	# features2()
	# titanic3()
	# cv_train()
	# modelensemble()
	titanic5()