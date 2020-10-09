# Import the necessary external libraries
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import accuracy_score

# Import any files in the root directory
import data


def train_rhc_nn(train_x, train_y, test_x, test_y):

	activations = ['identity', 'relu', 'sigmoid', 'tanh']
	colors = ['g', 'b', 'r', 'c']

	legend_items = [mlines.Line2D([0], [0], color='k', linestyle='-', markersize=5, label='Train'),
					mlines.Line2D([], [], color='k', linestyle=':', markersize=5, label='Test'),
					mlines.Line2D([0], [0], color='g', lw=2),
					mlines.Line2D([0], [0], color='b', lw=2),
					mlines.Line2D([0], [0], color='r', lw=2),
					mlines.Line2D([0], [0], color='c', lw=2)]

	legend_labels = ['Train', 'Test', 'identity', 'relu', 'sigmoid', 'tanh']



	for index, activation in enumerate(activations):
		training_curve = []
		testing_curve = []

		for i in range(1,1000,20):

			learning_rate = i / 1000

			nn_model = mlrose.NeuralNetwork(hidden_nodes = [8, 6], activation = activation, algorithm = 'random_hill_climb', max_iters = 1000, 
			 			bias = True, is_classifier = True, learning_rate = learning_rate, early_stopping = True, clip_max = 5, max_attempts = 100, random_state = 3)

			nn_model.fit(train_x, train_y)

			# Assess accuracy on training set
			y_train_pred = nn_model.predict(train_x)
			y_train_accuracy = accuracy_score(train_y, y_train_pred)
			training_curve.append({'learning_rate': learning_rate, 'accuracy': y_train_accuracy})

			# Assess accuracy on test set
			y_test_pred = nn_model.predict(test_x)
			y_test_accuracy = accuracy_score(test_y, y_test_pred)
			testing_curve.append({'learning_rate': learning_rate, 'accuracy': y_test_accuracy})

		plt.plot([x['learning_rate'] for x in training_curve], [y['accuracy'] for y in training_curve], linestyle='-', c=colors[index], label=activation)
		plt.plot([x['learning_rate'] for x in testing_curve], [y['accuracy'] for y in testing_curve], linestyle=':', c=colors[index], label=activation)

	plt.title("NN Training: RHC")
	plt.xlabel("Learning Rate")
	plt.ylabel("Accuracy")
	plt.legend(legend_items, legend_labels, loc="lower right")
	plt.tight_layout()
	plt.savefig("nn_rhc")
	plt.cla()


def train_sa_nn(train_x, train_y, test_x, test_y):

	activations = ['identity', 'relu', 'sigmoid', 'tanh']
	markers = ['o', 'x', '^', 's']
	colors = ['g', 'b', 'r', 'c']

	legend_items = [mlines.Line2D([], [], color='k', marker='o', markersize=5, linestyle='None', label='identity'),
					mlines.Line2D([], [], color='k', marker='x', markersize=5, linestyle='None', label='relu'),
					mlines.Line2D([], [], color='k', marker='^', markersize=5, linestyle='None', label='sigmoid'),
					mlines.Line2D([], [], color='k', marker='s', markersize=5, linestyle='None', label='tanh'),
					mlines.Line2D([0], [0], color='g', lw=2),
					mlines.Line2D([0], [0], color='b', lw=2),
					mlines.Line2D([0], [0], color='r', lw=2),
					mlines.Line2D([0], [0], color='c', lw=2)]

	legend_labels = ['identity', 'relu', 'sigmoid', 'tanh', 'geom', 'arith', 'exp']

	schedules = [
		{"title": "Geometric", "schedule": mlrose.GeomDecay(init_temp=10, decay=0.95, min_temp=1), "abbrev": "geom"},
		{"title": "Arithmetic", "schedule": mlrose.ArithDecay(init_temp=10, decay=0.95, min_temp=1), "abbrev": "arith"},
		{"title": "Exponential", "schedule": mlrose.ExpDecay(init_temp=10, exp_const=0.05, min_temp=1), "abbrev": "exp"}
	]

	for index, activation in enumerate(activations):

		for jindex, schedule in enumerate(schedules):


			nn_model = mlrose.NeuralNetwork(hidden_nodes=[8, 6, 5], activation=activation, algorithm='simulated_annealing', schedule=schedule['schedule'], max_iters=1000, 
			 			bias=True, is_classifier=True, learning_rate=0.0001, early_stopping=True, clip_max=5, max_attempts=100, random_state=3)

			fitted_model = nn_model.fit(train_x, train_y)

			# Assess accuracy on test set
			y_test_pred = nn_model.predict(test_x)

			plt.plot(fitted_model.loss, accuracy_score(test_y, y_test_pred), linestyle='-', c=colors[jindex], label=activation, marker=markers[index])

	plt.title("NN Training: SA \n Decay Schedules")
	plt.xlabel("Loss")
	plt.ylabel("Accuracy")
	plt.legend(legend_items, legend_labels, loc="lower right")
	plt.tight_layout()
	plt.savefig("nn_sa_by_schedule")
	plt.cla()

def nn_sa_tanh_training(train_x, train_y, test_x, test_y):

	curve = []

	for i in range(1,100,10):
		exp_const = i / 100

		schedule = mlrose.ExpDecay(init_temp=10, exp_const=exp_const, min_temp=1)

		nn_model = mlrose.NeuralNetwork(hidden_nodes=[8, 6, 5], activation='tanh', algorithm='simulated_annealing', schedule=schedule, max_iters=1000, 
			 			bias=True, is_classifier=True, learning_rate=0.0001, early_stopping=True, clip_max=5, max_attempts=100, random_state=3)

		train_start = time.time()
		fitted_model = nn_model.fit(train_x, train_y)
		train_duration = time.time() - train_start

		# Assess accuracy on test set
		pred_start = time.time()
		y_test_pred = nn_model.predict(test_x)
		pred_duration = time.time() - pred_start

		curve.append({"loss": fitted_model.loss, "exp_const": exp_const, "train": train_duration, "pred": pred_duration})

		
	fig, ax1 = plt.subplots()

	ax1.set_xlabel("Exponential Constant")
	ax1.set_ylabel("Loss", c="r")
	ax1.plot([x['exp_const'] for x in curve],[y['loss'] for y in curve], label="Training Loss", c="r")
	ax1.tick_params(axis='y', labelcolor="r")

	ax2 = ax1.twinx()

	ax2.set_ylabel('Duration', c="b")
	ax2.plot([x['exp_const'] for x in curve],[y['train'] for y in curve], label="Training", c="b", linestyle=":")
	ax2.plot([x['exp_const'] for x in curve],[y['pred'] for y in curve], label="Testing", c="b", linestyle="-")
	ax2.tick_params(axis="y", labelcolor='b')

	plt.title("NN Training: SA \n Exp Decay")
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig("nn_sa_exp_const")
	plt.cla()


 
def rhc_nn_loss_by_activation(train_x, train_y, test_x, test_y):

	activations = ['identity', 'relu', 'sigmoid', 'tanh']
	markers = ['o', 'x', '^', 's']
	colors = ['g', 'b', 'r', 'c']

	legend_items = [mlines.Line2D([], [], color='k', marker='o', markersize=5, linestyle='None', label='identity'),
					mlines.Line2D([], [], color='k', marker='x', markersize=5, linestyle='None', label='relu'),
					mlines.Line2D([], [], color='k', marker='^', markersize=5, linestyle='None', label='sigmoid'),
					mlines.Line2D([], [], color='k', marker='s', markersize=5, linestyle='None', label='tanh'),
					mlines.Line2D([0], [0], color='g', lw=2, label='rhc'),
					# mlines.Line2D([0], [0], color='b', lw=2),
					# mlines.Line2D([0], [0], color='r', lw=2),
					# mlines.Line2D([0], [0], color='c', lw=2)
					]

	legend_labels = ['identity', 'relu', 'sigmoid', 'tanh', 'RHC']


	for index, activation in enumerate(activations):

		nn_model = mlrose.NeuralNetwork(hidden_nodes = [8, 6], activation = activation, algorithm = 'random_hill_climb', max_iters = 1000, 
			 			bias = True, is_classifier = True, learning_rate = 0.001, early_stopping = True, clip_max = 5, max_attempts = 100, random_state = 3)

		fitted_model = nn_model.fit(train_x, train_y)

		y_train_pred = nn_model.predict(train_x)

		plt.plot(fitted_model.loss, accuracy_score(train_y, y_train_pred), marker=markers[index], label=activation, c=colors[0])

	plt.title("NN Training: RHC")
	plt.xlabel("Loss")
	plt.ylabel("Accuracy")
	plt.legend(legend_items, legend_labels, loc="lower right")
	plt.tight_layout()
	plt.savefig("nn_loss_vs_accuracy")
	plt.cla()


def nn_ga(train_x, train_y, test_x, test_y):

	activations = ['identity', 'relu', 'sigmoid', 'tanh']
	markers = ['o', 'x', '^', 's']
	colors = ['g', 'b', 'r', 'c', 'm', 'k']
	pop_sizes = [50, 200]
	mutation_probs = [0.1, 0.5]

	legend_items = [
					mlines.Line2D([0], [0], color='g', lw=2),
					mlines.Line2D([0], [0], color='b', lw=2),
					mlines.Line2D([0], [0], color='r', lw=2),
					mlines.Line2D([0], [0], color='c', lw=2),
					mlines.Line2D([0], [0], color='m', lw=2),
					mlines.Line2D([0], [0], color='k', lw=2)]

	legend_labels = ['pop 50, prob 0.1', 'pop 200, prob 0.1', 'pop 50, prob 0.5', 'pop 200, prob 0.5']

	counter = 0

	for index, mutation_prob in enumerate(mutation_probs):
		for jindex, pop_size in enumerate(pop_sizes):

			nn_model = mlrose.NeuralNetwork(hidden_nodes= [8, 6], activation='tanh', algorithm='genetic_alg', pop_size=pop_size, mutation_prob=mutation_prob, max_iters=1000, 
			 			bias=True, is_classifier=True, learning_rate=0.0001, early_stopping=True, clip_max=1000, max_attempts=100, random_state=3)

			fitted_model = nn_model.fit(train_x, train_y)

			plt.plot(fitted_model.fitness_curve, c=colors[counter])

			counter += 1


	plt.title("NN Training: GA")
	plt.xlabel("Fitness")
	plt.ylabel("Accuracy")
	plt.legend(legend_items, legend_labels, loc="lower right")
	plt.tight_layout()
	plt.savefig("nn_ga")
	plt.cla()

def nn_sa_temps(train_x, train_y, test_x, test_y):


	min_temps = [1, 5]
	init_temps = [100, 500]
	markers = ['o', 'x', '^', 's']
	colors = ['g', 'b', 'r', 'c', 'y', 'm']

	counter = 0

	for index, min_temp in enumerate(min_temps):

		for jindex, init_temp in enumerate(init_temps):

			schedule = mlrose.ExpDecay(init_temp=init_temp, exp_const=0.001, min_temp=min_temp)

			nn_model = mlrose.NeuralNetwork(hidden_nodes=[8, 6], activation='tanh', algorithm='simulated_annealing', schedule=schedule, max_iters=1000, 
				 			bias=True, is_classifier=True, learning_rate=0.0001, early_stopping=True, clip_max=1000, max_attempts=100, random_state=3, curve=True)


			fitted_model = nn_model.fit(train_x, train_y)

			label = "min_temp: " + str(min_temp) + ", init_temp: " + str(init_temp)

			plt.plot(fitted_model.fitness_curve, c=colors[counter], label=label)


			counter += 1

	
	plt.xlabel("No. of Iterations")
	plt.ylabel("Fitness")
	plt.title("NN Training: SA")
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig("nn_sa_temps")
	plt.cla()



def combined_nn():
	''' This is a function to combine the three NNs generated in this process and compare them. '''

	rhc = mlrose.NeuralNetwork(hidden_nodes = [8, 6, 5], activation = 'tanh', algorithm = 'random_hill_climb', max_iters = 1000, 
			 			bias = True, is_classifier = True, learning_rate = 0.001, early_stopping = True, clip_max = 5, max_attempts = 100, random_state = 3)

	schedule = mlrose.ExpDecay(init_temp=10, exp_const=0.01, min_temp=1)

	sa = mlrose.NeuralNetwork(hidden_nodes=[8, 6, 5], activation='tanh', algorithm='simulated_annealing', schedule=schedule, max_iters=1000, 
			 			bias=True, is_classifier=True, learning_rate=0.0001, early_stopping=True, clip_max=5, max_attempts=100, random_state=3)

	# ga = 


if __name__ == '__main__':
	train_x, train_y, test_x, test_y = data.main('adult_data.csv')

	# train_rhc_nn(train_x, train_y, test_x, test_y)
	# print(accuracy_score(test_y, y_test_pred))

	# rhc_nn_loss_by_activation(train_x, train_y, test_x, test_y)

	# train_sa_nn(train_x, train_y, test_x, test_y)
	# nn_sa_tanh_training(train_x, train_y, test_x, test_y)

	# nn_sa_temps(train_x, train_y, test_x, test_y)

	nn_ga(train_x, train_y, test_x, test_y)








