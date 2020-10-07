# This is the reader for the csv to parse the dataset to be used for the analysis
# Zach Sirera - Fall 2020

# Dataset details: 
# Name: Wine Quality Dataset

import csv
import random

def main(filename):
	data = get_all_data(filename)
	training_data, testing_data = separate_data_fixed(0.8, data)
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

def separate_data_fixed(fraction, data):

	training_data = []
	testing_data = []

	for each in data:
		if len(training_data) / len(data) <= fraction:
			training_data.append(each)
		else:
			testing_data.append(each)

	return training_data, testing_data


def parse(dataset):
	''' This is a function to parse the dataset dictionary and format it properly for implementation '''

	workclass = {'Private': 1, 'Self-emp-not-inc': 2, 'Self-emp-inc': 3, 'Federal-gov': 4, 'Local-gov': 5, 'State-gov': 6, 'Without-pay': 7, 'Never-worked': 8}
	education = {'Bachelors': 1, 'Some-college': 2, '11th': 3, 'HS-grad': 4, 'Prof-school': 5, 'Assoc-acdm': 6, 'Assoc-voc': 7, '9th': 8, '7th-8th': 9, '12th': 10, 'Masters': 11, '1st-4th': 12, '10th': 13, 'Doctorate': 14, '5th-6th': 15, 'Preschool': 16}
	marital_status = {'Married-civ-spouse': 1, 'Divorced': 2, 'Never-married': 3, 'Separated': 4, 'Widowed': 5, 'Married-spouse-absent': 6, 'Married-AF-spouse': 7}
	occupation = {'Tech-support': 1, 'Craft-repair': 2, 'Other-service': 3, 'Sales': 4, 'Exec-managerial': 5, 'Prof-specialty': 7, 'Handlers-cleaners': 8, 'Machine-op-inspct': 9, 'Adm-clerical': 10, 'Farming-fishing': 11, 'Transport-moving': 12, 'Priv-house-serv': 13, 'Protective-serv': 14, 'Armed-Forces': 15}
	relationship = {'Wife': 1, 'Own-child': 2, 'Husband': 3,'Not-in-family': 4, 'Other-relative': 5, 'Unmarried': 6}
	race = {'White': 1, 'Asian-Pac-Islander': 2, 'Amer-Indian-Eskimo': 3, 'Other': 4, 'Black': 5}
	native_country = {'United-States': 1, 'Cambodia': 2, 'England': 3, 'Puerto-Rico': 4, 'Canada': 5, 'Germany': 6, 'Outlying-US(Guam-USVI-etc)': 7, 'India': 8, 'Japan': 9, 'Greece': 10, 'South': 11, 'China': 12, 'Cuba': 13, 'Iran': 14, 'Honduras': 15, 'Philippines': 16, 'Italy': 17, 'Poland': 18, 'Jamaica': 19, 'Vietnam': 20, 'Mexico': 21, 'Portugal': 22, 'Ireland': 22, 'France': 23, 'Dominican-Republic': 24, 'Laos':25, 'Ecuador': 26, 'Taiwan': 27, 'Haiti': 28, 'Columbia': 29, 'Hungary': 30, 'Guatemala': 31, 'Nicaragua': 32, 'Scotland': 33, 'Thailand': 34, 'Yugoslavia': 35, 'El-Salvador': 36, 'Trinadad&Tobago': 37, 'Peru': 38, 'Hong': 39, 'Holand-Netherlands': 40}
	sex = {'Female': 1, 'Male': 2}
	income = {'>50K': 1, '<=50K': 2}



	x = []
	y = []

	for row in dataset:
		if ' ?' in row.values(): 
			pass
		else:
			x.append([
				int(row['age']),
				workclass[row['workclass'].lstrip()],
				float(row['fnlwgt']),
				education[row['education'].lstrip()],
				float(row['education-num']),
				marital_status[row['marital-status'].lstrip()],
				occupation[row['occupation'].lstrip()],
				relationship[row['relationship'].lstrip()],
				race[row['race'].lstrip()],
				sex[row['sex'].lstrip()],
				float(row['capital-gain']),
				float(row['capital-loss']),
				float(row['hours-per-week']),
				native_country[row['native-country'].lstrip()]
			])
			y.append(income[row['income'].lstrip()])

	return x, y








