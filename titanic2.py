#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'preston.zhu'

import numpy as np
import pandas as pd
import re
import operator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor

import pdb

def get_title(name):
	# 正则表达式搜索，搜索到如' Mr.'的字符串
	title_search = re.search(' ([A-Za-z]+)\.', name)
	# 搜索到则非空
	if title_search:
		# group()返回一个字符串，字符串[0]为空格，故从[1]开始读取
		return title_search.group(1)
	return ""

family_id_mapping = {}
def get_family_id(row):
	last_name = row['Name'].split(',')[0]
	family_id = "{0}{1}".format(last_name, row['FamilySize'])

	if family_id not in family_id_mapping:
		if len(family_id_mapping) == 0:
			current_id = 1
		else:
			# operator.itemgetter(1)即取出对象第一个域的内容
			# dict.items(): dict所有元素成都表示成元组，然后展开成list
			# max()函数按key参数比较大小，取出第一个列表中最大的对象
			current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
		# 姓氏名映射成数值家庭号
		family_id_mapping[family_id] = current_id
	return family_id_mapping[family_id]

# 函数根据年龄和性别分成三类
def get_person(passenger):
	age, sex = passenger
	if age < 14:	# child age define 14
		return 'child'
	elif sex == 'female':
		return 'female_adult'
	else:
		return 'male_adult'

# 获取家族名称
def process_surname(nm):
	return nm.split(',')[0].lower()

perishing_female_surnames = []
def perishing_mother_wife(passenger):
	surname, Pclass, person = passenger
	# 筛选“成年女性，过逝的，有同行家庭成员的”，并返回1，否则返回0
	return 1.0 if (surname in perishing_female_surnames) else 0.0

surviving_male_surnames = []
def surviving_father_husband(passenger):
	surname, Pclass, person = passenger
	return 1.0 if (surname in surviving_male_surnames) else 0.0

"""
特征工程：
采用将训练集与测试集的数据拼接在一起，然后进行回归，再补充'Age'的缺失值，这是一个很好的方法；
移除掉'Ticket'特征
'Embarked'特征采用众数'S'补充缺失值
'Fare'采用中间值补充缺件值
增加'TitleCat'特征：从名称中抽取表示个人身份地位的称为来表示
增加'CabinCat'特征：先将缺件值补充字符'0'，然后提取第一个字符做为其分类。缺失值太多，另作为一个分类
增加'EmbarkedCat'特征：由'Cabin'特征转换成数值分类表示
增加'Sex_male'和'Sex_female'两个特征：由'Sex'特征仿拟(dummy)
增加'FamilySize'特征：由'SibSp'和'Parch'两个特征之各，表示同行家人数量
增加'NameLength'特征：由名称字符数量表示。
增加'FamilyId'特征：先提取'Name'特征中的姓氏，并按字母排序编号得出。另外，同行家人少于3人的，'FamilyId'统一
				   归于-1类
增加'person'特征：由'Age'和'Sex'特征，小于14岁定义为儿童'child'，大于14岁的女性定义为成年女性
                 'female_adult'，大于14岁的男性定义为成年男性'male_adult'
增加'persion_child', 'person_female_adult', 'person_male_adult'三个特征：由'person'特征仿拟(dummy)
增加'surname'特征：由'Name'特征提取出姓氏部分
增加'perishing_mother_wife'特征：过逝的母亲或妻子，对家人的存活影响会比较大
增加‘surviving_father_husband'特征：存活的父亲或丈夫，对家人的存活影响也会比较大
最后选择进行训练的特征为：
	'Age', 'Fare', 'Parch', 'Pclass', 'SibSp','male_adult', 'female_adult', 'child',
	'perishing_mother_wife', 'surviving_father_husband', 'TitleCat', 'CabinCat',
	'Sex_female', 'Sex_male', 'EmbarkedCat', 'FamilySize', 'NameLength', 'FamilyId'

由于经过拼接，所以需要对训练集与测试集进行拆分，前891个实例为训练集，后418个实例为测试集
"""
def features():
	train_data = pd.read_csv("input/train.csv", dtype={"Age": np.float64})
	test_data = pd.read_csv("input/test.csv", dtype={"Age": np.float64})

	# 按列方向连接两个DataFrame，test_data排在train_data之后
	combined2 = pd.concat([train_data, test_data], axis=0)

	# 去掉'Ticket'特征，axis=1表示每行，inplace=True表示直接作用于本身
	combined2.drop(['Ticket'], axis=1, inplace=True)

	# Embarked特征使用众数补充缺失值
	# inplace参数为True，函数作用于当前变量之上
	combined2.Embarked.fillna('S', inplace=True)
	# Fare特征使用中间值补充缺失值
	combined2.Fare.fillna(combined2.Fare[combined2.Fare.notnull()].median(), inplace=True)

	# 新建'Title'特征，从'Name'特征中提取称谓来表示
	# Series.apply(func)对Series中每一个元素都调用一次func函数
	combined2['Title'] = combined2["Name"].apply(get_title)
	title_mapping = {
		"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7,
		"Col": 7, "Mlle": 8, "Mme": 8, "Don": 7, "Dona": 10, "Lady": 10,
		"Countess": 10, "Jonkheer": 10, "Sir": 7, "Capt": 7, "Ms": 2
	}
	# 新建'TitleCat'特征，则'Title'映射成数值
	# map()函数使用dict对'Title'每个元素都进行映射
	combined2["TitleCat"] = combined2.loc[:, 'Title'].map(title_mapping)
	# 新建'CabinCat'特征。缺失值补充为0，其它项取第一个字母
	# Categorical先统计数组中有多少个不同项，按升序排列，然后用数值表示各个分类
	combined2["CabinCat"] = pd.Categorical(combined2.Cabin.fillna('0').apply(lambda x: x[0])).codes
	# Cabin特征缺失值为'0'填充
	combined2.Cabin.fillna('0', inplace=True)
	# 以数值表示'Embarked'各个分类
	combined2['EmbarkedCat'] = pd.Categorical(combined2.Embarked).codes

	# 延行方向进行连接，创建新DataFrame；增加两个性别相关列，并将'Survived'特征移动至最后一列
	# pandas.get_dummies()将'Sex'特征重构成'Sex_male'和'Sex_female'两列，
	# 'Sex_male'列中，male表示为1，female表示为0. 'Sex_female'则相反
	full_data = pd.concat([
		combined2.drop(['Survived'], axis=1),
		pd.get_dummies(combined2.Sex, prefix='Sex'),
		combined2.Survived
	], axis=1)

	# 新建'FamilySize'特征，使用'SibSp'和'Parch'两个家庭成员特征求和得出
	full_data['FamilySize'] = full_data['SibSp'] + full_data['Parch']
	# 新建'NameLength'特征，使用名称长度表示
	full_data['NameLength'] = full_data.Name.apply(lambda x: len(x))

	family_ids = full_data.apply(get_family_id, axis=1)
	# 将所有家庭成员人数小于3个的设置成-1类，归成一类
	family_ids[full_data['FamilySize'] < 3] = -1
	# 新建'FamilyId'特征
	full_data['FamilyId'] = family_ids

	# 追加'person'特征列，'person'特征由年龄和性别划分成child, femal_adult, male_adult
	full_data = pd.concat([
		full_data,
		pd.DataFrame(full_data[['Age', 'Sex']].apply(get_person, axis=1), columns=['person'])
	], axis=1)
	# dummies person
	dummies = pd.get_dummies(full_data['person'])
	# 追加'persion_child', 'person_female_adult', 'person_male_adult'三个特征
	full_data = pd.concat([full_data, dummies], axis=1)

	# 新建姓氏名称'surname'特征
	full_data['surname'] = full_data['Name'].apply(process_surname)

	# 筛选“成年女性，过逝的，有同行家庭成员的”，去重
	perishing_female_surnames = list(set(full_data[
		(full_data.female_adult == 1.0)
		& (full_data.Survived == 0.0)
		& ((full_data.Parch > 0) | (full_data.SibSp > 0))]['surname'].values))

	# 新建'perishing_mother_wife'特征，如果是“已过逝且有同行家人的成年女性”为1，否则为0
	full_data['perishing_mother_wife'] \
		= full_data[['surname', 'Pclass', 'person']].apply(perishing_mother_wife, axis=1)

	# 筛选“成年男性，存活的，有同行家人了”，去重
	surviving_male_surnames = list(set(full_data[
		(full_data.male_adult == 1.0)
		& (full_data.Survived == 1.0)
		& ((full_data.Parch > 0) | (full_data.SibSp > 0))]['surname']))

	full_data['surviving_father_husband'] \
		= full_data[['surname', 'Pclass', 'person']].apply(surviving_father_husband, axis=1)

	# 定义筛选器
	classers = [
		'Fare', 'Parch', 'Pclass', 'SibSp', 'TitleCat', 'CabinCat', 'Sex_female', 'Sex_male',
		'EmbarkedCat', 'FamilySize', 'NameLength', 'FamilyId'
	]
	# ExtraTreesRegressor模型，用带'Age'特征的数据回归出'Age'特征缺失的值
	age_et = ExtraTreesRegressor(n_estimators=200)
	# 筛选'Age'不为空的数据作为训练集
	X_train = full_data.loc[full_data.Age.notnull(), classers]
	# 筛选'Age'不为空的数据的'Age'特征作为训练集的结果标签
	Y_train = full_data.loc[full_data.Age.notnull(), ['Age']]
	# 'Age'为空即为测试集
	X_test = full_data.loc[full_data.Age.isnull(), classers]
	# np.ravel()转换为np.array
	age_et.fit(X_train, np.ravel(Y_train))
	age_preds = age_et.predict(X_test)
	# 将回归预测的结果，填充到原数据集中
	full_data.loc[full_data.Age.isnull(), ['Age']] = age_preds

	# 定义筛选器
	model_dummys = [
		'Age', 'Fare', 'Parch', 'Pclass', 'SibSp','male_adult', 'female_adult', 'child',
		'perishing_mother_wife', 'surviving_father_husband', 'TitleCat', 'CabinCat',
		'Sex_female', 'Sex_male', 'EmbarkedCat', 'FamilySize', 'NameLength', 'FamilyId'
	]

	# 筛选出训练集，测试集
	X_data = full_data.iloc[:891, :]
	X_train = X_data.loc[:, model_dummys]
	Y_data = full_data.iloc[:891, :]
	y_train = Y_data.loc[:, ['Survived']]
	X_t_data = full_data.iloc[891:, :]
	X_test = X_t_data.loc[:, model_dummys]
	test_PassengerId = X_t_data.PassengerId.as_matrix()

	return X_train, y_train, X_test, test_PassengerId

def titanic():
	print('Preparing Data...')

	X_train, y_train, X_test, test_PassengerId = features()

	print('Train RandomForestClassifier Model...')
	# 随机森林模型
	model_rf = RandomForestClassifier(n_estimators=300,
									  min_samples_leaf=4,
									  class_weight={0:0.745,1:0.255})
	# 训练
	model_rf.fit(X_train, np.ravel(y_train))

	print('Predictings...')

	model_results = model_rf.predict(X_test)

	print('Generate Submission File...')

	submission = pd.DataFrame({
		'PassengerId': test_PassengerId,
		'Survived': model_results.astype(np.int32)
	})
	submission.to_csv('prediction7.csv', index=False)

	print('Done.')

if __name__ == '__main__':
	titanic()