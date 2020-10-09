# This is the reader for the csv to parse the dataset to be used for the analysis
# Zach Sirera - Fall 2020

# Dataset details: 
# Name: Wine Quality Dataset

import csv
import random
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd

def main(filename):
	# data = get_all_data(filename)
	df = load_data()
	df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income'])
	training_data, testing_data = separate_data_fixed(df, 0.8)
	train_x, train_y = parse(training_data)
	test_x, test_y = parse(testing_data)

	return train_x, train_y, test_x, test_y

def get_all_data(filename):
	''' This function simply reads the csv into a dictionary which will then be parsed by the various learners depending on what 
	their individual input rules are. '''

	all_data = []

	with open(filename, newline='') as csvfile:
		reader = csv.reader(csvfile, delimiter=',')
		for row in reader:
			all_data.append({
				'age': row[0],
				'workclass': row[1],
				'fnlwgt': row[2],
				'education': row[3],
				'education-num': row[4],
				'marital-status': row[5],
				'occupation': row[6],
				'relationship': row[7],
				'race': row[8],
				'sex': row[9],
				'capital-gain': row[10],
				'capital-loss': row[11],
				'hours-per-week': row[12],
				'native-country': row[13],
				'income': row[14],
				})

	return all_data

def separate_data_rand(fraction, data):
	''' This is a function to generate a set of indices at random to serve as training data and the rest will be test data '''

	training_data = []
	testing_data = []

	for each in data:
		if random.random() <= fraction:
			training_data.append(each)
		else:
			testing_data.append(each)

	return training_data, testing_data

# def separate_data_fixed(fraction, data):

# 	training_data = []
# 	testing_data = []

# 	for each in data:
# 		if len(training_data) / len(data) <= fraction:
# 			training_data.append(each)
# 		else:
# 			testing_data.append(each)

# 	return training_data, testing_data

def separate_data_fixed(df, fraction):


	msk = np.random.rand(len(df)) < fraction
	train = df[msk]
	test = df[~msk]

	return train, test


# def parse(dataset):
# 	''' This is a function to parse the dataset dictionary and format it properly for implementation '''

# 	workclass = {'Private': [1, 0, 0, 0, 0, 0, 0, 0], 'Self-emp-not-inc': [0, 1, 0, 0, 0, 0, 0, 0], 'Self-emp-inc': [0, 0, 1, 0, 0, 0, 0, 0], 'Federal-gov': [0, 0, 0, 1, 0, 0, 0, 0], 'Local-gov': [0, 0, 0, 0, 1, 0, 0, 0], 'State-gov': [0, 0, 0, 0, 0, 1, 0, 0], 'Without-pay': [0, 0, 0, 0, 0, 0, 1, 0], 'Never-worked': [0, 0, 0, 0, 0, 0, 0, 1]}
# 	education = {'Bachelors': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Some-college': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], '11th': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'HS-grad': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Prof-school': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Assoc-acdm': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Assoc-voc': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], '9th': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], '7th-8th': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], '12th': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'Masters': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], '1st-4th': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], '10th': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'Doctorate': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], '5th-6th': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'Preschool': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
# 	marital_status = {'Married-civ-spouse': [1, 0, 0, 0, 0, 0, 0], 'Divorced': [0, 1, 0, 0, 0, 0, 0], 'Never-married': [0, 0, 1, 0, 0, 0, 0], 'Separated': [0, 0, 0, 1, 0, 0, 0], 'Widowed': [0, 0, 0, 0, 1, 0, 0], 'Married-spouse-absent': [0, 0, 0, 0, 0, 1, 0], 'Married-AF-spouse': [0, 0, 0, 0, 0, 0, 1]}
# 	occupation = {'Tech-support': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Craft-repair': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Other-service': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Sales': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Exec-managerial': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Prof-specialty': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'Handlers-cleaners': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'Machine-op-inspct': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'Adm-clerical': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'Farming-fishing': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'Transport-moving': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'Priv-house-serv': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'Protective-serv': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'Armed-Forces': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
# 	relationship = {'Wife': [1, 0, 0, 0, 0, 0], 'Own-child': [0, 1, 0, 0, 0, 0], 'Husband': [0, 0, 1, 0, 0, 0],'Not-in-family': [0, 0, 0, 1, 0, 0], 'Other-relative': [0, 0, 0, 0, 1, 0], 'Unmarried': [0, 0, 0, 0, 0, 1]}
# 	race = {'White': [1, 0, 0, 0, 0], 'Asian-Pac-Islander': [0, 1, 0, 0, 0], 'Amer-Indian-Eskimo': [0, 0, 1, 0, 0], 'Other': [0, 0, 0, 1, 0], 'Black': [0, 0, 0, 0, 1]}
# 	native_country = {'United-States': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Cambodia': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'England': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Puerto-Rico': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Canada': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Germany': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Outlying-US(Guam-USVI-etc)': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'India': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Japan': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Greece': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'South': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'China': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Cuba': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Iran': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Honduras': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Philippines': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Italy': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Poland': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Jamaica': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Vietnam': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Mexico': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Portugal': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Ireland': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'France': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Dominican-Republic': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Laos': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Ecuador': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Taiwan': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Haiti': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Columbia': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Hungary': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Guatemala': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'Nicaragua': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'Scotland': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 'Thailand': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'Yugoslavia': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], 'El-Salvador': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'Trinadad&Tobago': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], 'Peru': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'Hong': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 'Holand-Netherlands': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
# 	sex = {'Female': [1, 0], 'Male': [0, 1]}
# 	income = {'>50K': [1, 0], '<=50K': [0, 1]}

# 	[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# 	x = []
# 	y = []

# 	for row in dataset:
# 		if ' ?' in row.values(): 
# 			pass
# 		else:
# 			x.append([
# 				int(row['age']),
# 				workclass[row['workclass'].lstrip()],
# 				float(row['fnlwgt']), #
# 				education[row['education'].lstrip()],
# 				float(row['education-num']),#
# 				marital_status[row['marital-status'].lstrip()],
# 				occupation[row['occupation'].lstrip()],
# 				relationship[row['relationship'].lstrip()],
# 				race[row['race'].lstrip()],
# 				sex[row['sex'].lstrip()],
# 				float(row['capital-gain']), #
# 				float(row['capital-loss']), #
# 				float(row['hours-per-week']), #
# 				native_country[row['native-country'].lstrip()]
# 			])
# 			y.append(income[row['income'].lstrip()])

# 	return x, y

def parse(df):

	return df.iloc[:, :108], df.iloc[:, 109:]


def load_data():
    return pd.read_csv('adult_data.csv')

# if __name__ == '__main__':
# 	df = load_data()
# 	df = pd.get_dummies(df, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'income'])
# 	# df_x, df_y = parse(df)
# 	train_df, test_df = separate_data_fixed(df, 0.8)

# 	print(train_df.shape)
# 	print(test_df.shape)






