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
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.neural_network import MLPClassifier

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
			 			bias=True, is_classifier=True, learning_rate=0.0001, early_stopping=True, clip_max=1000, max_attempts=100, random_state=3, curve=True)

			fitted_model = nn_model.fit(train_x, train_y)

			plt.plot(fitted_model.fitness_curve, c=colors[counter])

			counter += 1


	plt.title("NN Training: GA")
	plt.xlabel("No. of Iterations")
	plt.ylabel("Fitness")
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



def combined_nn(train_x, train_y, test_x, test_y):
	''' This is a function to combine the three NNs generated in this process and compare them. '''

	markers = ['o', 'x', '^', 's']
	colors = ['g', 'b', 'r', 'c']
	learning_rates = [0.0001, 0.01, 0.1]



	plot_1 = plt.figure(1)

	for index, learning_rate in enumerate(learning_rates):

		# Random Hill Climb
		rhc = mlrose.NeuralNetwork(hidden_nodes = [8, 6], activation = 'tanh', algorithm = 'random_hill_climb', max_iters = 1000, 
				 			bias = True, is_classifier = True, learning_rate = learning_rate, early_stopping = True, clip_max = 5, max_attempts = 100, random_state = 3, curve=True)
		rhc_start = time.time()	
		trained_rhc = rhc.fit(train_x, train_y)
		rhc_duration = time.time() - rhc_start


		# Get data for duration vs accuracy curve
		rhc_test_pred = trained_rhc.predict(test_x)
		rhc_test_accuracy = accuracy_score(test_y, rhc_test_pred)
		plt.plot(rhc_duration, rhc_test_accuracy, marker=markers[index], color=colors[0], label="RHC")


		# Get data for ROC
		rhc_probs = trained_rhc.predicted_probs
		rhc_fpr, rhc_tpr, rhc_threshold = roc_curve(test_y, rhc_probs, pos_label=1)
		rhc_roc_auc = auc(rhc_fpr, rhc_tpr)
		rhc_label = "RHC: AUC = " + str(round(rhc_roc_auc, 3))



		# Simulated Annealing
		schedule = mlrose.ExpDecay(init_temp=10, exp_const=0.01, min_temp=1)
		sa = mlrose.NeuralNetwork(hidden_nodes=[8, 6], activation='tanh', algorithm='simulated_annealing', schedule=schedule, max_iters=1000, 
				 			bias=True, is_classifier=True, learning_rate=learning_rate, early_stopping=True, clip_max=5, max_attempts=100, random_state=3, curve = True)

		sa_start = time.time()
		trained_sa = sa.fit(train_x, train_y)
		sa_duration = time.time() - sa_start

		# Get data for duration vs accuracy curve
		sa_test_pred = trained_sa.predict(test_x)
		sa_test_accuracy = accuracy_score(test_y, sa_test_pred)
		plt.plot(sa_duration, sa_test_accuracy, marker=markers[index], color=colors[1], label="SA")

		# Get data for ROC curve
		sa_probs = trained_sa.predicted_probs
		sa_fpr, sa_tpr, sa_threshold = roc_curve(test_y, sa_probs, pos_label=1)
		sa_roc_auc = auc(sa_fpr, sa_tpr)
		sa_label = "SA: AUC = " + str(round(sa_roc_auc, 3))


		# Genetic Algorithm
		ga = mlrose.NeuralNetwork(hidden_nodes= [8, 6], activation='tanh', algorithm='genetic_alg', pop_size=200, mutation_prob=0.1, max_iters=1000, 
				 			bias=True, is_classifier=True, learning_rate=learning_rate, early_stopping=True, clip_max=1000, max_attempts=100, random_state=3, curve=True)

		ga_start = time.time()
		trained_ga = ga.fit(train_x, train_y)
		ga_duration = time.time() - ga_start

		# Get data for duration vs accuracy curve
		ga_test_pred = trained_ga.predict(test_x)
		ga_test_accuracy = accuracy_score(test_y, ga_test_pred)
		plt.plot(ga_duration, ga_test_accuracy, marker=markers[index], color=colors[2], label="GA")

		# Get data for ROC curve
		ga_probs = trained_ga.predicted_probs
		ga_fpr, ga_tpr, ga_threshold = roc_curve(test_y, ga_probs, pos_label=1)
		ga_roc_auc = auc(ga_fpr, ga_tpr)
		ga_label = "GA: AUC = " + str(round(ga_roc_auc, 3))


		# Assignment 1's Neural Network 
		old = MLPClassifier(solver='sgd', alpha=1e-5, learning_rate_init=learning_rate, activation='tanh', hidden_layer_sizes=(8, 6), random_state=1)

		old_start = time.time()
		trained_old = old.fit(train_x, np.ravel(train_y))
		old_duration = time.time() - old_start

		# Get data for duration vs accuracy curve
		old_test_pred = trained_old.predict(test_x)
		old_test_accuracy = accuracy_score(test_y, old_test_pred)
		plt.plot(old_duration, old_test_accuracy, marker=markers[index], color=colors[3], label="A1")

		# Get data for ROC Curve
		old_probs = []

		for index, each in test_x.iterrows():
			prob = old.predict_proba([each])
			old_probs.append(prob[:,1][0])

		old_fpr, old_tpr, old_threshold = roc_curve(test_y, old_probs, pos_label=1)
		old_roc_auc = auc(old_fpr, old_tpr)
		old_label = "A1: AUC = " + str(round(old_roc_auc, 3))

	
	legend_items = [mlines.Line2D([], [], color='k', marker='o', markersize=5, linestyle='None'),
					mlines.Line2D([], [], color='k', marker='x', markersize=5, linestyle='None'),
					mlines.Line2D([], [], color='k', marker='^', markersize=5, linestyle='None'),
					mlines.Line2D([0], [0], color='g', lw=2),
					mlines.Line2D([0], [0], color='b', lw=2),
					mlines.Line2D([0], [0], color='r', lw=2),
					mlines.Line2D([0], [0], color='c', lw=2)]

	legend_labels = ['0.0001', '0.01', '0.1', 'RHC', 'SA', 'GA', 'A1']




	# Plot the duration vs accuracy chart
	plt.xlabel("Training Duration (s)")
	plt.xscale("log")
	plt.ylabel("Test Accuracy")
	plt.title("Neural Networks \n Learning Rate")
	plt.legend(legend_items, legend_labels, loc="upper right")
	plt.tight_layout()
	plt.savefig("nn_combined_duration")
	plt.cla()



	# # Plot the ROC chart
	plot_2 = plt.figure(2)

	plt.plot(rhc_fpr, rhc_tpr, label=rhc_label)
	plt.plot(old_fpr, old_tpr, label=old_label)
	plt.plot(sa_fpr, sa_tpr, label=sa_label)
	plt.plot(ga_fpr, ga_tpr, label=ga_label)

	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	plt.title("Neural Networks")
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig("nn_rocs")
	plt.cla()


def combined_fitness_curves(train_x, train_y, test_x, test_y):

	markers = ['o', 'x', '^', 's']
	colors = ['g', 'b', 'r', 'c']




	# Random Hill Climb
	rhc = mlrose.NeuralNetwork(hidden_nodes = [8, 6], activation = 'tanh', algorithm = 'random_hill_climb', max_iters = 1000, 
			 			bias = True, is_classifier = True, learning_rate = 0.01, early_stopping = True, clip_max = 5, max_attempts = 100, random_state = 3, curve=True)
	
	trained_rhc = rhc.fit(train_x, train_y)
	plt.plot(trained_rhc.fitness_curve, label="RHC")
	



	# Simulated Annealing
	schedule = mlrose.ExpDecay(init_temp=10, exp_const=0.01, min_temp=1)
	sa = mlrose.NeuralNetwork(hidden_nodes=[8, 6], activation='tanh', algorithm='simulated_annealing', schedule=schedule, max_iters=1000, 
			 			bias=True, is_classifier=True, learning_rate=0.01, early_stopping=True, clip_max=5, max_attempts=100, random_state=3, curve = True)

	trained_sa = sa.fit(train_x, train_y)
	plt.plot(trained_sa.fitness_curve, label="SA")




	# Genetic Algorithm
	ga = mlrose.NeuralNetwork(hidden_nodes= [8, 6], activation='tanh', algorithm='genetic_alg', pop_size=200, mutation_prob=0.1, max_iters=1000, 
			 			bias=True, is_classifier=True, learning_rate=0.01, early_stopping=True, clip_max=1000, max_attempts=100, random_state=3, curve=True)

	trained_ga = ga.fit(train_x, train_y)
	plt.plot(trained_ga.fitness_curve, label="SA")



	# # Assignment 1's Neural Network 
	# old = MLPClassifier(solver='sgd', alpha=1e-5, learning_rate_init=learning_rate, activation='tanh', hidden_layer_sizes=(8, 6), random_state=1)

	# trained_old = old.fit(train_x, np.ravel(train_y))

	
	



	# Plot the duration vs accuracy chart
	plt.xlabel("No of Iterations")
	plt.ylabel("Fitness")
	plt.title("Neural Networks")
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig("nn_combined_fitness")
	plt.cla()



def combined_learning_curves(train_x, train_y, test_x, test_y):

	rhc_accuracy = []
	sa_accuracy = []
	ga_accuracy = []

	for i in range(1, 1000, 50):

		# Random Hill Climb
		rhc = mlrose.NeuralNetwork(hidden_nodes = [8, 6], activation = 'tanh', algorithm = 'random_hill_climb', max_iters = 1000, 
				 			bias = True, is_classifier = True, learning_rate = 0.01, early_stopping = True, clip_max = 5, max_attempts = 100, random_state = 3, curve=True)
		
		trained_rhc = rhc.fit(train_x, train_y)

		rhc_train_pred = trained_rhc.predict(train_x)
		rhc_train_accuracy = accuracy_score(train_y, rhc_train_pred)

		rhc_test_pred = trained_rhc.predict(test_x)
		rhc_test_accuracy = accuracy_score(test_y, rhc_test_pred)

		rhc_accuracy.append({"i": i, "train": rhc_train_accuracy, "test": rhc_test_accuracy})


		# Simulated Annealing
		schedule = mlrose.ExpDecay(init_temp=10, exp_const=0.01, min_temp=1)
		sa = mlrose.NeuralNetwork(hidden_nodes=[8, 6], activation='tanh', algorithm='simulated_annealing', schedule=schedule, max_iters=1000, 
				 			bias=True, is_classifier=True, learning_rate=0.01, early_stopping=True, clip_max=5, max_attempts=100, random_state=3, curve = True)

		trained_sa = sa.fit(train_x, train_y)

		sa_train_pred = trained_sa.predict(train_x)
		sa_train_accuracy = accuracy_score(train_y, sa_train_pred)

		sa_test_pred = trained_sa.predict(test_x)
		sa_test_accuracy = accuracy_score(test_y, sa_test_pred)

		sa_accuracy.append({"i": i, "train": sa_train_accuracy, "test": sa_test_accuracy})




		# Genetic Algorithm
		ga = mlrose.NeuralNetwork(hidden_nodes= [8, 6], activation='tanh', algorithm='genetic_alg', pop_size=200, mutation_prob=0.1, max_iters=1000, 
				 			bias=True, is_classifier=True, learning_rate=0.01, early_stopping=True, clip_max=1000, max_attempts=100, random_state=3, curve=True)

		trained_ga = ga.fit(train_x, train_y)
		plt.plot(trained_ga.fitness_curve, label="SA")

		ga_train_pred = trained_ga.predict(train_x)
		ga_train_accuracy = accuracy_score(train_y, ga_train_pred)

		ga_test_pred = trained_ga.predict(test_x)
		ga_test_accuracy = accuracy_score(test_y, ga_test_pred)

		ga_accuracy.append({"i": i, "train": ga_train_accuracy, "test": ga_test_accuracy})


	# Plot Training curves
	plt.plot([x['i'] for x in rhc_accuracy], [y['train'] for y in rhc_accuracy], linestyle=":", color='r')
	plt.plot([x['i'] for x in sa_accuracy], [y['train'] for y in sa_accuracy], linestyle=":", color='g')
	plt.plot([x['i'] for x in ga_accuracy], [y['train'] for y in ga_accuracy], linestyle=":", color='b')


	# Plot Testing curves
	plt.plot([x['i'] for x in rhc_accuracy], [y['test'] for y in rhc_accuracy], linestyle="-", color='r')
	plt.plot([x['i'] for x in sa_accuracy], [y['test'] for y in sa_accuracy], linestyle="-", color='g')
	plt.plot([x['i'] for x in ga_accuracy], [y['test'] for y in ga_accuracy], linestyle="-", color='b')

	legend_items = [
				mlines.Line2D([0], [0], color='r', lw=2),
				mlines.Line2D([0], [0], color='g', lw=2),
				mlines.Line2D([0], [0], color='b', lw=2),
				mlines.Line2D([0], [0], color='k', lw=2, linestyle=":"),
				mlines.Line2D([0], [0], color='k', lw=2, linestyle="-")
				]

	legend_labels = ['RHC', 'SA', 'GA', 'Train', 'Test']



	# Plot the duration vs accuracy chart
	plt.xlabel("No. of Iterations")
	plt.ylabel("Accuracy")
	plt.title("Neural Networks: Learning Curves")
	plt.legend(legend_items, legend_labels, loc="upper right")
	plt.tight_layout()
	plt.savefig("nn_combined_learning_curves")
	plt.cla()





if __name__ == '__main__':

	train_x, train_y, test_x, test_y = data.main('adult_data.csv')

	# train_rhc_nn(train_x, train_y, test_x, test_y)
	# rhc_nn_loss_by_activation(train_x, train_y, test_x, test_y)


	# train_sa_nn(train_x, train_y, test_x, test_y)
	# nn_sa_tanh_training(train_x, train_y, test_x, test_y)
	# nn_sa_temps(train_x, train_y, test_x, test_y)

 

	# nn_ga(train_x, train_y, test_x, test_y)

	# combined_nn(train_x, train_y, test_x, test_y)
	# combined_fitness_curves(train_x, train_y, test_x, test_y)
	# combined_learning_curves(train_x, train_y, test_x, test_y)

	pass








