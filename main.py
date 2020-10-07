# This is a submission for GT 7641 Machine Learning - Assignment 2: Randomized Optimization
# Zach Sirera - Fall 2020

# Import the necessary external libraries
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import time

# Import files from root directory




def get_salesman_coords(length):
	''' Used once to generate a uniform random layout ''' 
	my_coords = []
	for i in range(length):
		my_coords.append((np.random.randint(0,100),(np.random.randint(0,100))))

	return my_coords


def get_6_peaks_state(length):

	return [np.random.randint(0,2) for i in range(length)]


def salesman():

	problem_name = "Travelling Salesman"

	models = [{"title": "Simulated Annealing", "abbrev": 'sa'}, 
				{"title": "Random Hill Climb", "abbrev": 'rhc'}, 
				{"title": "Genetic Algorithm", "abbrev": 'ga'}, 
				{"title": "MIMIC", "abbrev": 'mimic'}]

	# from get_salesman_coords
	coords_big = [(73, 75), (22, 65), (84, 71), (63, 57), (66, 57), (92, 48), (57, 18), (91, 7), (67, 90), (57, 52), (0, 85), 
	(10, 34), (50, 54), (2, 9), (39, 17), (94, 18), (80, 55), (94, 15), (95, 61), (59, 59), (38, 14), (16, 72), (97, 98), (28, 81), 
	(8, 76), (74, 0), (70, 82), (47, 58), (58, 25), (91, 31), (29, 62), (13, 59), (29, 21), (16, 7), (7, 90), (54, 63), (51, 84), 
	(0, 54), (64, 87), (10, 98), (69, 86), (10, 38), (10, 23), (1, 93), (3, 52), (3, 1), (87, 95), (82, 9), (76, 73), (58, 18), 
	(89, 54), (93, 35), (86, 32), (67, 41), (70, 98), (60, 24), (63, 81), (19, 27), (75, 62), (34, 32), (56, 13), (84, 61), (54, 13), 
	(59, 38), (96, 5), (55, 78), (52, 59), (35, 10), (66, 36), (40, 39), (65, 27), (48, 24), (13, 87), (10, 53), (85, 93), (36, 94), 
	(4, 8), (3, 9), (12, 68), (25, 54), (51, 32), (85, 5), (33, 73), (26, 88), (27, 46), (54, 82), (3, 26), (94, 64), (16, 63), (76, 88), 
	(27, 71), (6, 57), (41, 48), (17, 72), (82, 30), (22, 66), (56, 29), (98, 75), (1, 81), (52, 16)]

	coords_small = [(7, 57), (74, 6), (18, 25), (56, 71), (5, 31), (21, 32), (47, 50), (75, 87), (44, 45), (68, 16)]

	coords_medium = [(44, 48), (94, 52), (4, 99), (83, 93), (58, 72), (31, 21), (11, 45), (87, 33), (4, 88), (8, 36), (78, 11), (55, 21), 
	(11, 31), (50, 14), (45, 77), (36, 36), (74, 92), (32, 88), (71, 37), (42, 62), (61, 71), (6, 71), (26, 54), (16, 95), (21, 54)]

	total_coords = [coords_small, coords_medium, coords_big] # , coords_medium, coords_big


	fitness_fn = mlrose.TravellingSales(coords=coords_big)
	optim = mlrose.TSPOpt(len(coords), fitness_fn=fitness_fn)

	colors = ['b', 'r', 'g', 'k', 'c', 'm']

	######### Generate map plots for all models
	for model in models: 
		
		fitness_scores = []
		times = []

		sa_attempts = [1, 10, 100, 1000, 10000, 100000] #1, 10, 100, 1000, 10000, 100000
		m_attempts = [1, 10, 100]
		rhc_restarts = [1, 10, 100, 1000] #1, 10, 100, 1000, 10000, 100000
		ga_attempts = [1, 10, 100, 1000, 10000] #1, 10, 100, 1000, 10000, 100000

		if model["abbrev"] == "sa":

			for index, attempt in enumerate(sa_attempts):

				time_start = time.time()
				best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(optim, curve = True, max_attempts = attempt)
				time_end = time.time()
				times.append({"max_attempts": attempt, "duration": round(time_end - time_start, 3)})

				fitness_scores.append(best_fitness)

				coords_ordered = [coords[each] for each in best_state]

				plt.plot([coord[0] for coord in coords_ordered], [coord[1] for coord in coords_ordered], c='b', lw=0.5)
				plt.scatter([coord[0] for coord in coords], [coord[1] for coord in coords], c='r')

				if attempt == 1:
					plt.title(str(model["title"]) + ": Travelling Salesman " + str(attempt) + " Attempt \n Fitness: " + str(round(best_fitness,2)))
				else: 
					plt.title(str(model["title"]) + ": Travelling Salesman " + str(attempt) + " Attempts \n Fitness: " + str(round(best_fitness,2)))

				plt.savefig("ts_" + str(model["abbrev"]) + "_" + str(attempt))
				plt.cla()

		if model['abbrev'] == "rhc":

			for index, restart in enumerate(rhc_restarts):

				time_start = time.time()
				best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(optim, curve=True, restarts=restart)
				time_end = time.time()
				times.append({"restarts": restart, "duration": round(time_end - time_start, 3)})

				fitness_scores.append(best_fitness)
				
				coords_ordered = [coords[each] for each in best_state]

				plt.plot([coord[0] for coord in coords_ordered], [coord[1] for coord in coords_ordered], c='b', lw=0.5)
				plt.scatter([coord[0] for coord in coords], [coord[1] for coord in coords], c='r')

				if restart == 1:
					plt.title(str(model["title"]) + ": Travelling Salesman " + str(restart) + " Restart \n Fitness: " + str(round(best_fitness,2)))
				else: 
					plt.title(str(model["title"]) + ": Travelling Salesman " + str(restart) + " Restarts \n Fitness: " + str(round(best_fitness,2)))

				plt.savefig("ts_" + str(model["abbrev"]) + "_" + str(restart))
				plt.cla()

		if model['abbrev'] == "ga":

			for index, attempt in enumerate(ga_attempts):
				
				time_start = time.time()
				best_state, best_fitness, fitness_curve = mlrose.genetic_alg(optim, pop_size=10, curve=True, max_attempts=attempt)
				time_end = time.time()
				times.append({"max_attempts": attempt, "duration": round(time_end - time_start, 3)})

				fitness_scores.append(best_fitness)

				coords_ordered = [coords[each] for each in best_state]

				plt.plot([coord[0] for coord in coords_ordered], [coord[1] for coord in coords_ordered], c='b', lw=0.5)
				plt.scatter([coord[0] for coord in coords], [coord[1] for coord in coords], c='r')

				if attempt == 1:
					plt.title(str(model["title"]) + ": Travelling Salesman " + str(attempt) + " Attempt \n Fitness: " + str(round(best_fitness,2)))
				else: 
					plt.title(str(model["title"]) + ": Travelling Salesman " + str(attempt) + " Attempts \n Fitness: " + str(round(best_fitness,2)))

				plt.savefig("ts_" + str(model["abbrev"]) + "_" + str(attempt))
				plt.cla()


			performance = []
			for i in range(10,200,10):
				start = time.time()
				best_state, best_fitness, fitness_curve = mlrose.genetic_alg(optim, pop_size=i, curve=True, max_attempts=10)

				performance.append({"fitness": best_fitness, "pop_size": i, "duration": time.time() - start})

			fig, ax1 = plt.subplots()
			ax1.set_xlabel('Population Size')
			ax1.set_ylabel("Fitness", c="r")
			ax1.plot([each["pop_size"] for each in performance], [each["fitness"] for each in performance], c="r")
			ax1.tick_params(axis='y', labelcolor="r")

			ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

			ax2.set_ylabel("Execution Time (s)", c="b")  # we already handled the x-label with ax1
			ax2.plot([each["pop_size"] for each in performance], [each["duration"] for each in performance], c="b")
			ax2.tick_params(axis='y', labelcolor="b")

			plt.title("Genetic Algorithm: Travelling Salesman \n Performance: 10 Attempts")

			fig.tight_layout()  # otherwise the right y-label is slightly clipped
			plt.savefig("ts_" + str(model["abbrev"]) + "_pop_size")
			plt.cla()

		if model['abbrev'] == 'mimic':

			for index, attempt in enumerate(m_attempts):

				time_start = time.time()
				best_state, best_fitness, fitness_curve = mlrose.mimic(optim, curve=True, max_attempts=attempt)
				time_end = time.time()
				times.append({"max_attempts": attempt, "duration": round(time_end - time_start, 3)})

				fitness_scores.append(best_fitness)
				
				coords_ordered = [coords[each] for each in best_state]

				plt.plot([coord[0] for coord in coords_ordered], [coord[1] for coord in coords_ordered], c='b', lw=0.5)
				plt.scatter([coord[0] for coord in coords], [coord[1] for coord in coords], c='r')

				if attempt == 1:
					plt.title(str(model["title"]) + ": Travelling Salesman " + str(attempt) + " Attempt \n Fitness: " + str(round(best_fitness,2)))
				else: 
					plt.title(str(model["title"]) + ": Travelling Salesman " + str(attempt) + " Attempts \n Fitness: " + str(round(best_fitness,2)))

				plt.savefig("ts_" + str(model["abbrev"]) + "_" + str(attempt))
				plt.cla()

		print(str(model['title']) + ": Durations")
		print(times)

		plt.plot(fitness_curve)
		plt.title(str(model["title"]) + ": Travelling Salesman \n " + str(attempts[-1]) + " Attempts")
		plt.ylabel('Fitness')
		plt.xlabel('No. of Iterations')
		plt.xscale("log")
		plt.savefig("ts_" + str(model["abbrev"]) + "_fitness_curve")
		plt.cla()

		plt.plot(attempts, [-score for score in fitness_scores])
		plt.title(str(model["title"]) + ": Travelling Salesman")
		plt.ylabel('Fitness')
		plt.xlabel('Max Attempts')
		plt.xscale("log")
		plt.savefig("ts_" + str(model["abbrev"]) + "_fitness_scores")
		plt.cla()



	######## Generate combined fitness curve chart
	schedule = mlrose.ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)
	sa_state, sa_fitness, sa_curve = mlrose.simulated_annealing(optim, curve = True, max_attempts=1000, schedule=schedule)
	rhc_state, rhc_fitness, rhc_curve = mlrose.random_hill_climb(optim, curve=True, restarts=10)
	ga_state, ga_fitness, ga_curve = mlrose.genetic_alg(optim, pop_size=630, mutation_prob=0.5, curve=True, max_attempts=20)
	m_state, m_fitness, m_curve = mlrose.mimic(optim, curve=True, max_attempts=6, keep_pct=0.2, pop_size=700)
	plt.plot(sa_curve, c="r", lw="0.5", label="Simmulated Annealing")
	plt.plot(ga_curve, c="b", lw="0.5", label="Genetic Alg")
	plt.plot(rhc_curve, c="g", lw="0.5", label="Random Hill Climb")
	plt.plot(m_curve, c="c", lw="0.5", label="MIMIC")
	plt.title("Travelling Salesman \n 100 Attempts")
	plt.xlabel("No. of Iterations")
	plt.ylabel("Fitness")
	plt.xscale("log")
	plt.legend(loc="lower right")
	plt.savefig("ts_fitness_combined")
	plt.cla()




	######## Generate combined fitness curve chart for multiple sizes

	markers = ['o', 'x', '^']


	for index, coords in enumerate(total_coords):

		sa_fitness_curve = []

		fitness_fn = mlrose.TravellingSales(coords=coords)	
		optim = mlrose.TSPOpt(len(coords), fitness_fn=fitness_fn)

		schedule = mlrose.ExpDecay(init_temp=10, exp_const=0.05, min_temp=1)

		sa_time_start = time.time()
		sa_state, sa_fitness, sa_curve = mlrose.simulated_annealing(optim, curve = True, max_attempts=10, schedule=schedule)
		sa_duration = time.time() - sa_time_start

		sa_fitness_curve.append({"time": sa_duration, "fitness": sa_fitness})

		plt.plot([x['time'] for x in sa_fitness_curve],[-y['fitness'] for y in sa_fitness_curve], lw=0.5, color='g', marker=markers[index])


	for index, coords in enumerate(total_coords):

		fitness_fn = mlrose.TravellingSales(coords=coords)	
		optim = mlrose.TSPOpt(len(coords), fitness_fn=fitness_fn)

		rhc_fitness_curve = []

		rhc_time_start = time.time()
		rhc_state, rhc_fitness, rhc_curve = mlrose.random_hill_climb(optim, curve=True, restarts=10)
		rhc_duration = time.time() - rhc_time_start

		rhc_fitness_curve.append({"time": rhc_duration, "fitness": rhc_fitness})

		plt.plot([x['time'] for x in rhc_fitness_curve],[-y['fitness'] for y in rhc_fitness_curve], linestyle='-', lw=0.5, color='b', marker=markers[index])

	for index, coords in enumerate(total_coords):

		fitness_fn = mlrose.TravellingSales(coords=coords)	
		optim = mlrose.TSPOpt(len(coords), fitness_fn=fitness_fn)

		ga_fitness_curve = []

		ga_time_start = time.time()
		ga_state, ga_fitness, ga_curve = mlrose.genetic_alg(optim, pop_size=630, mutation_prob=0.5, curve=True, max_attempts=10)
		ga_duration = time.time() - ga_time_start

		ga_fitness_curve.append({"time": ga_duration, "fitness": ga_fitness})

		plt.plot([x['time'] for x in ga_fitness_curve],[-y['fitness'] for y in ga_fitness_curve], label="MIMIC", lw=0.5, color='r', marker=markers[index])

	for index, coords in enumerate(total_coords):


		fitness_fn = mlrose.TravellingSales(coords=coords)	
		optim = mlrose.TSPOpt(len(coords), fitness_fn=fitness_fn)

		m_fitness_curve = []

		m_time_start = time.time()
		m_state, m_fitness, m_curve = mlrose.mimic(optim, curve=True, max_attempts=10, keep_pct=0.2, pop_size=700)
		m_duration = time.time() - m_time_start

		m_fitness_curve.append({"time": m_duration, "fitness": m_fitness})

		plt.plot([x['time'] for x in m_fitness_curve],[-y['fitness'] for y in m_fitness_curve], label="MIMIC", lw=0.5, color='c', marker=markers[index])



	legend_items = [mlines.Line2D([], [], color='k', marker='o', linestyle='None', markersize=5, label='n 10'),
		mlines.Line2D([], [], color='k', marker='x', linestyle='None', markersize=5, label='n 25'),
		mlines.Line2D([], [], color='k', marker='^', linestyle='None', markersize=5, label='n 100'),
		mlines.Line2D([0], [0], color='g', lw=2),
		mlines.Line2D([0], [0], color='b', lw=2),
		mlines.Line2D([0], [0], color='r', lw=2),
		mlines.Line2D([0], [0], color='c', lw=2)]


	legend_labels = ['n 10', 'n 25', 'n 100', 'SA', 'RHC', 'GA', 'MIMIC']



	plt.title("Travelling Salesman \n Problem Size")
	plt.xlabel("Duration")
	plt.xscale("log")
	plt.ylabel("Fitness")
	plt.legend(legend_items, legend_labels, loc="lower right")
	plt.savefig("ts_duration_vs_fitness")
	plt.cla()


def ts_sa_decay_schedules(): 
	######## Compare decay schedules

	# from get_salesman_coords
	coords_big = [(73, 75), (22, 65), (84, 71), (63, 57), (66, 57), (92, 48), (57, 18), (91, 7), (67, 90), (57, 52), (0, 85), 
	(10, 34), (50, 54), (2, 9), (39, 17), (94, 18), (80, 55), (94, 15), (95, 61), (59, 59), (38, 14), (16, 72), (97, 98), (28, 81), 
	(8, 76), (74, 0), (70, 82), (47, 58), (58, 25), (91, 31), (29, 62), (13, 59), (29, 21), (16, 7), (7, 90), (54, 63), (51, 84), 
	(0, 54), (64, 87), (10, 98), (69, 86), (10, 38), (10, 23), (1, 93), (3, 52), (3, 1), (87, 95), (82, 9), (76, 73), (58, 18), 
	(89, 54), (93, 35), (86, 32), (67, 41), (70, 98), (60, 24), (63, 81), (19, 27), (75, 62), (34, 32), (56, 13), (84, 61), (54, 13), 
	(59, 38), (96, 5), (55, 78), (52, 59), (35, 10), (66, 36), (40, 39), (65, 27), (48, 24), (13, 87), (10, 53), (85, 93), (36, 94), 
	(4, 8), (3, 9), (12, 68), (25, 54), (51, 32), (85, 5), (33, 73), (26, 88), (27, 46), (54, 82), (3, 26), (94, 64), (16, 63), (76, 88), 
	(27, 71), (6, 57), (41, 48), (17, 72), (82, 30), (22, 66), (56, 29), (98, 75), (1, 81), (52, 16)]
	
	fitness_fn = mlrose.TravellingSales(coords=coords_big)
	optim = mlrose.TSPOpt(len(coords), fitness_fn=fitness_fn)

	schedules = [
	{"title": "Geometric", "schedule": mlrose.GeomDecay(init_temp=10, decay=0.95, min_temp=1), "abbrev": "geom"},
	{"title": "Arithmetic", "schedule": mlrose.ArithDecay(init_temp=10, decay=0.95, min_temp=1), "abbrev": "arith"},
	{"title": "Exponential", "schedule": mlrose.ExpDecay(init_temp=10, exp_const=0.05, min_temp=1), "abbrev": "exp"}]

	for index, schedule in enumerate(schedules):
		best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(optim, curve=True, max_attempts=100, schedule=schedule["schedule"])

		plt.plot(fitness_curve, c=colors[index], lw=0.5, label=str(schedule['title']))

	plt.title("Travelling Salesman: Simmulated Annealing \n Decay Schedules, 100 Attempts")
	plt.xlabel("No. of Iterations")
	plt.ylabel("Fitness")
	plt.legend(loc="lower right")
	plt.xscale("log")
	plt.tight_layout()
	plt.savefig("ts_sa_decay")
	plt.cla()

	## Best occurs at schedule = expDecay


def ts_sa_sched_params():
	# ######## Compare params for decay schedule 

	coords_big = [(73, 75), (22, 65), (84, 71), (63, 57), (66, 57), (92, 48), (57, 18), (91, 7), (67, 90), (57, 52), (0, 85), 
	(10, 34), (50, 54), (2, 9), (39, 17), (94, 18), (80, 55), (94, 15), (95, 61), (59, 59), (38, 14), (16, 72), (97, 98), (28, 81), 
	(8, 76), (74, 0), (70, 82), (47, 58), (58, 25), (91, 31), (29, 62), (13, 59), (29, 21), (16, 7), (7, 90), (54, 63), (51, 84), 
	(0, 54), (64, 87), (10, 98), (69, 86), (10, 38), (10, 23), (1, 93), (3, 52), (3, 1), (87, 95), (82, 9), (76, 73), (58, 18), 
	(89, 54), (93, 35), (86, 32), (67, 41), (70, 98), (60, 24), (63, 81), (19, 27), (75, 62), (34, 32), (56, 13), (84, 61), (54, 13), 
	(59, 38), (96, 5), (55, 78), (52, 59), (35, 10), (66, 36), (40, 39), (65, 27), (48, 24), (13, 87), (10, 53), (85, 93), (36, 94), 
	(4, 8), (3, 9), (12, 68), (25, 54), (51, 32), (85, 5), (33, 73), (26, 88), (27, 46), (54, 82), (3, 26), (94, 64), (16, 63), (76, 88), 
	(27, 71), (6, 57), (41, 48), (17, 72), (82, 30), (22, 66), (56, 29), (98, 75), (1, 81), (52, 16)]
	
	fitness_fn = mlrose.TravellingSales(coords=coords_big)
	optim = mlrose.TSPOpt(len(coords), fitness_fn=fitness_fn)

	fitness = []

	for i in range(10,1000,10):

		schedule = mlrose.ExpDecay(init_temp=10, exp_const=0.05, min_temp=(i/100))
		best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(optim, curve=True, max_attempts=10, schedule=schedule)
		fitness.append({"min_temp": (i/100), "fitness": best_fitness})

	plt.plot([x['min_temp'] for x in fitness],[-y['fitness'] for y in fitness])
	plt.title("Travelling Salesman: Simmulated Annealing \n Exp Decay, 10 Attempts")
	plt.xlabel("Min Temp")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("ts_sa_exp_min_temp")
	plt.cla()

	# #### Best occures at nowhere. Use default. 

	fitness_master = []

	for i in range(10):

		fitness = []

		for i in range(5,100,5):

			schedule = mlrose.ExpDecay(init_temp=10, exp_const=(i / 100), min_temp=1)
			best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(optim, curve=True, max_attempts=10, schedule=schedule)
			fitness.append({"exp_const": (i / 100), "fitness": best_fitness})

		fitness_master.append(fitness)

	for each in fitness_master:
		plt.scatter([x['exp_const'] for x in each],[y['fitness'] for y in each],c="b")

	plt.title("Travelling Salesman: Simmulated Annealing \n Exp Decay, 1000 Attempts")
	plt.xlabel("Exp Constant")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("ts_sa_exp_const")
	plt.cla()

	##### Best occurs at exp_const = no best. Use default. 

	

def ts_ga_pop_size():
	######### Compare population sizes for genetic algorithm

	coords_big = [(73, 75), (22, 65), (84, 71), (63, 57), (66, 57), (92, 48), (57, 18), (91, 7), (67, 90), (57, 52), (0, 85), 
	(10, 34), (50, 54), (2, 9), (39, 17), (94, 18), (80, 55), (94, 15), (95, 61), (59, 59), (38, 14), (16, 72), (97, 98), (28, 81), 
	(8, 76), (74, 0), (70, 82), (47, 58), (58, 25), (91, 31), (29, 62), (13, 59), (29, 21), (16, 7), (7, 90), (54, 63), (51, 84), 
	(0, 54), (64, 87), (10, 98), (69, 86), (10, 38), (10, 23), (1, 93), (3, 52), (3, 1), (87, 95), (82, 9), (76, 73), (58, 18), 
	(89, 54), (93, 35), (86, 32), (67, 41), (70, 98), (60, 24), (63, 81), (19, 27), (75, 62), (34, 32), (56, 13), (84, 61), (54, 13), 
	(59, 38), (96, 5), (55, 78), (52, 59), (35, 10), (66, 36), (40, 39), (65, 27), (48, 24), (13, 87), (10, 53), (85, 93), (36, 94), 
	(4, 8), (3, 9), (12, 68), (25, 54), (51, 32), (85, 5), (33, 73), (26, 88), (27, 46), (54, 82), (3, 26), (94, 64), (16, 63), (76, 88), 
	(27, 71), (6, 57), (41, 48), (17, 72), (82, 30), (22, 66), (56, 29), (98, 75), (1, 81), (52, 16)]
	
	fitness_fn = mlrose.TravellingSales(coords=coords_big)
	optim = mlrose.TSPOpt(len(coords), fitness_fn=fitness_fn)

	fitness = []

	for i in range(1,50,1):
		best_state, best_fitness, fitness_curve = mlrose.genetic_alg(optim, curve=True, max_attempts=10, pop_size=i)

		fitness.append({"pop_size": i, "fitness": best_fitness})

	plt.plot([x['pop_size'] for x in fitness], [-y['fitness'] for y in fitness])
	plt.title("Travelling Salesman: Genetic Algorithm")
	plt.xlabel("Population Size")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("ts_ga_pop_size")
	plt.cla()

	##### Best occurs at pop_size = 630


def ts_ga_mut_prob():

	coords_big = [(73, 75), (22, 65), (84, 71), (63, 57), (66, 57), (92, 48), (57, 18), (91, 7), (67, 90), (57, 52), (0, 85), 
	(10, 34), (50, 54), (2, 9), (39, 17), (94, 18), (80, 55), (94, 15), (95, 61), (59, 59), (38, 14), (16, 72), (97, 98), (28, 81), 
	(8, 76), (74, 0), (70, 82), (47, 58), (58, 25), (91, 31), (29, 62), (13, 59), (29, 21), (16, 7), (7, 90), (54, 63), (51, 84), 
	(0, 54), (64, 87), (10, 98), (69, 86), (10, 38), (10, 23), (1, 93), (3, 52), (3, 1), (87, 95), (82, 9), (76, 73), (58, 18), 
	(89, 54), (93, 35), (86, 32), (67, 41), (70, 98), (60, 24), (63, 81), (19, 27), (75, 62), (34, 32), (56, 13), (84, 61), (54, 13), 
	(59, 38), (96, 5), (55, 78), (52, 59), (35, 10), (66, 36), (40, 39), (65, 27), (48, 24), (13, 87), (10, 53), (85, 93), (36, 94), 
	(4, 8), (3, 9), (12, 68), (25, 54), (51, 32), (85, 5), (33, 73), (26, 88), (27, 46), (54, 82), (3, 26), (94, 64), (16, 63), (76, 88), 
	(27, 71), (6, 57), (41, 48), (17, 72), (82, 30), (22, 66), (56, 29), (98, 75), (1, 81), (52, 16)]
	
	fitness_fn = mlrose.TravellingSales(coords=coords_big)
	optim = mlrose.TSPOpt(len(coords), fitness_fn=fitness_fn)

	######### Compare mutation rates for genetic algorithm
	mutations = [0.05, 0.1, 0.2, 0.5]
	colors = ['r', 'b', 'g', 'k']

	for i, mutation in enumerate(mutations):
		best_state, best_fitness, fitness_curve = mlrose.genetic_alg(optim, curve=True, max_attempts=10, pop_size=630, mutation_prob=mutation)

		plt.plot(fitness_curve, lw=0.5, c=colors[i], label=str(mutation))

	plt.title("Travelling Salesman: Genetic Algorithm \n Mutation Probabilities")
	plt.xlabel("No. of Iterations")
	plt.legend(loc="lower right")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("ts_ga_mut_probs")
	plt.cla()


	##### Best occurs at mutation_prob 

def ts_m_pop_size():
	######### Compare population sizes for mimic

	coords_big = [(73, 75), (22, 65), (84, 71), (63, 57), (66, 57), (92, 48), (57, 18), (91, 7), (67, 90), (57, 52), (0, 85), 
	(10, 34), (50, 54), (2, 9), (39, 17), (94, 18), (80, 55), (94, 15), (95, 61), (59, 59), (38, 14), (16, 72), (97, 98), (28, 81), 
	(8, 76), (74, 0), (70, 82), (47, 58), (58, 25), (91, 31), (29, 62), (13, 59), (29, 21), (16, 7), (7, 90), (54, 63), (51, 84), 
	(0, 54), (64, 87), (10, 98), (69, 86), (10, 38), (10, 23), (1, 93), (3, 52), (3, 1), (87, 95), (82, 9), (76, 73), (58, 18), 
	(89, 54), (93, 35), (86, 32), (67, 41), (70, 98), (60, 24), (63, 81), (19, 27), (75, 62), (34, 32), (56, 13), (84, 61), (54, 13), 
	(59, 38), (96, 5), (55, 78), (52, 59), (35, 10), (66, 36), (40, 39), (65, 27), (48, 24), (13, 87), (10, 53), (85, 93), (36, 94), 
	(4, 8), (3, 9), (12, 68), (25, 54), (51, 32), (85, 5), (33, 73), (26, 88), (27, 46), (54, 82), (3, 26), (94, 64), (16, 63), (76, 88), 
	(27, 71), (6, 57), (41, 48), (17, 72), (82, 30), (22, 66), (56, 29), (98, 75), (1, 81), (52, 16)]
	
	fitness_fn = mlrose.TravellingSales(coords=coords_big)
	optim = mlrose.TSPOpt(len(coords), fitness_fn=fitness_fn)

	fitness = []

	for i in range(1000,1200,100):
		best_state, best_fitness, fitness_curve = mlrose.mimic(optim, curve=True, max_attempts=3, pop_size=i)

		fitness.append({"pop_size": i, "fitness": best_fitness})

	plt.plot([x['pop_size'] for x in fitness], [-y['fitness'] for y in fitness])
	plt.title("Travelling Salesman: MIMIC")
	plt.xlabel("Population Size")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("ts_mimic_pop_size_ctd")
	plt.cla()


def ts_m_keep_pct():
	######### Compare mutation rates for mimic

	coords_big = [(73, 75), (22, 65), (84, 71), (63, 57), (66, 57), (92, 48), (57, 18), (91, 7), (67, 90), (57, 52), (0, 85), 
	(10, 34), (50, 54), (2, 9), (39, 17), (94, 18), (80, 55), (94, 15), (95, 61), (59, 59), (38, 14), (16, 72), (97, 98), (28, 81), 
	(8, 76), (74, 0), (70, 82), (47, 58), (58, 25), (91, 31), (29, 62), (13, 59), (29, 21), (16, 7), (7, 90), (54, 63), (51, 84), 
	(0, 54), (64, 87), (10, 98), (69, 86), (10, 38), (10, 23), (1, 93), (3, 52), (3, 1), (87, 95), (82, 9), (76, 73), (58, 18), 
	(89, 54), (93, 35), (86, 32), (67, 41), (70, 98), (60, 24), (63, 81), (19, 27), (75, 62), (34, 32), (56, 13), (84, 61), (54, 13), 
	(59, 38), (96, 5), (55, 78), (52, 59), (35, 10), (66, 36), (40, 39), (65, 27), (48, 24), (13, 87), (10, 53), (85, 93), (36, 94), 
	(4, 8), (3, 9), (12, 68), (25, 54), (51, 32), (85, 5), (33, 73), (26, 88), (27, 46), (54, 82), (3, 26), (94, 64), (16, 63), (76, 88), 
	(27, 71), (6, 57), (41, 48), (17, 72), (82, 30), (22, 66), (56, 29), (98, 75), (1, 81), (52, 16)]
	
	fitness_fn = mlrose.TravellingSales(coords=coords_big)
	optim = mlrose.TSPOpt(len(coords), fitness_fn=fitness_fn)

	mutations = [0.05, 0.1, 0.2, 0.5]
	colors = ['r', 'b', 'g', 'k']

	for i, mutation in enumerate(mutations):
		best_state, best_fitness, fitness_curve = mlrose.mimic(optim, curve=True, max_attempts=3, pop_size=700, keep_pct=mutation)

		plt.plot(fitness_curve, lw=0.5, c=colors[i], label=str(mutation))

	plt.title("Travelling Salesman: MIMIC \n Keep Probabilities")
	plt.xlabel("No. of Iterations")
	plt.ylabel("Fitness")
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig("ts_mimic_keep_probs")
	plt.cla()

	###### Best occurs at keep_pct=0.02



def four_peaks_problem_size_duration():
	######## Generate combined fitness curve chart using results from feature sweep for different problem sizes 


	sizes = [25, 50, 100]
	markers = ['o', 'x', '^']
	legend_items = [mlines.Line2D([], [], color='k', marker='o', linestyle='None', markersize=5, label='n 25'),
		mlines.Line2D([], [], color='k', marker='x', linestyle='None', markersize=5, label='n 50'),
		mlines.Line2D([], [], color='k', marker='^', linestyle='None', markersize=5, label='n 100'),
		mlines.Line2D([0], [0], color='g', lw=2),
		mlines.Line2D([0], [0], color='b', lw=2),
		mlines.Line2D([0], [0], color='r', lw=2),
		mlines.Line2D([0], [0], color='c', lw=2)]

	legend_labels = ['n 25', 'n 50', 'n 100', 'SA', 'RHC', 'GA', 'MIMIC']

	for index, size in enumerate(sizes):

		fitness = mlrose.FourPeaks()
		optim = mlrose.DiscreteOpt(size, fitness, maximize=True)
		schedule = mlrose.ExpDecay(init_temp=10, exp_const=0.55, min_temp=3)
		
		start = time.time()
		sa_state, sa_fitness, sa_curve = mlrose.simulated_annealing(optim, curve=True, max_attempts=10, schedule=schedule)
		duration = time.time() - start

		plt.plot(duration,sa_fitness, lw=0.5, color='g', marker=markers[index])


	for index, size in enumerate(sizes):

		fitness = mlrose.FourPeaks()
		optim = mlrose.DiscreteOpt(size, fitness, maximize=True)
		
		start = time.time()
		rhc_state, rhc_fitness, rhc_curve = mlrose.random_hill_climb(optim, curve=True, restarts=100)
		duration = time.time() - start

		plt.plot(duration,rhc_fitness, lw=0.5, color='b', marker=markers[index])


	for index, size in enumerate(sizes):

		fitness = mlrose.FourPeaks()
		optim = mlrose.DiscreteOpt(size, fitness, maximize=True)
		
		start = time.time()
		ga_state, ga_fitness, ga_curve = mlrose.genetic_alg(optim, curve=True, max_attempts=100, pop_size=3, mutation_prob=0.05)		
		duration = time.time() - start

		plt.plot(duration,ga_fitness, lw=0.5, color='r', marker=markers[index])

	for index, size in enumerate(sizes):

		fitness = mlrose.FourPeaks()
		optim = mlrose.DiscreteOpt(size, fitness, maximize=True)
		
		start = time.time()
		m_state, m_fitness, m_curve = mlrose.mimic(optim, curve=True, max_attempts=100, pop_size=5, keep_pct=0.5)		
		duration = time.time() - start

		plt.plot(duration,m_fitness, lw=0.5, color='c', marker=markers[index])


	plt.title("Four Peaks \n Problem Size")
	plt.xlabel("Duration")
	plt.xscale("log")
	plt.ylabel("Fitness")
	plt.legend(legend_items, legend_labels, loc="lower right")
	plt.savefig("4p_duration_vs_fitness")
	plt.cla()



def four_peaks():

	problem_name = "Four Peaks"

	models = [{"title": "Simulated Annealing", "abbrev": 'sa'}, 
				{"title": "Random Hill Climb", "abbrev": 'rhc'}, 
				{"title": "Genetic Algorithm", "abbrev": 'ga'}, 
				{"title": "MIMIC", "abbrev": 'mimic'}]


	fitness = mlrose.FourPeaks()
	optim = mlrose.DiscreteOpt(25, fitness, maximize=True)

	colors = ['b', 'r', 'g', 'k', 'c', 'm']

	# ######## Generate combined fitness curve chart using results from feature sweep


	# schedule = mlrose.ExpDecay(init_temp=10, exp_const=0.55, min_temp=3)

	sa_state, sa_fitness, sa_curve = mlrose.simulated_annealing(optim, curve=True, max_attempts=10, schedule=schedule)
	rhc_state, rhc_fitness, rhc_curve = mlrose.random_hill_climb(optim, curve=True, restarts=100)
	ga_state, ga_fitness, ga_curve = mlrose.genetic_alg(optim, curve=True, max_attempts=100, pop_size=3, mutation_prob=0.05)
	m_state, m_fitness, m_curve = mlrose.mimic(optim, curve=True, max_attempts=100, pop_size=5, keep_pct=0.5)
	plt.plot(sa_curve, c="r", lw="0.5", label="Simmulated Annealing")
	plt.plot(ga_curve, c="b", lw="0.5", label="Genetic Alg")
	plt.plot(rhc_curve, c="g", lw="0.5", label="Random Hill Climb")
	plt.plot(m_curve, c="c", lw="0.5", label="MIMIC")
	plt.title("Four Peaks")
	plt.xlabel("No. of Iterations")
	plt.ylabel("Fitness")
	plt.xscale("log")
	plt.legend(loc="lower right")
	plt.savefig("4p_fitness_combined")
	plt.cla()


	



def four_peaks_decay_schedules():

	problem_name = "Four Peaks"

	models = [{"title": "Simulated Annealing", "abbrev": 'sa'}, 
				{"title": "Random Hill Climb", "abbrev": 'rhc'}, 
				{"title": "Genetic Algorithm", "abbrev": 'ga'}, 
				{"title": "MIMIC", "abbrev": 'mimic'}]


	fitness = mlrose.FourPeaks()
	optim = mlrose.DiscreteOpt(25, fitness, maximize=True)

	colors = ['b', 'r', 'g', 'k', 'c', 'm']

	######## Compare decay schedules
	schedules = [
	{"title": "Geometric", "schedule": mlrose.GeomDecay(init_temp=10, decay=0.95, min_temp=1), "abbrev": "geom"},
	{"title": "Arithmetic", "schedule": mlrose.ArithDecay(init_temp=10, decay=0.95, min_temp=1), "abbrev": "arith"},
	{"title": "Exponential", "schedule": mlrose.ExpDecay(init_temp=10, exp_const=0.05, min_temp=1), "abbrev": "exp"}]

	for index, schedule in enumerate(schedules):
		best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(optim, curve=True, max_attempts=100, schedule=schedule["schedule"])

		plt.plot(fitness_curve, c=colors[index], lw=0.5, label=str(schedule['title']))

	plt.title("Four Peaks: Simmulated Annealing \n Decay Schedules, 100 Attempts")
	plt.xlabel("No. of Iterations")
	plt.ylabel("Fitness")
	plt.legend(loc="lower right")
	plt.xscale("log")
	plt.tight_layout()
	plt.savefig("4p_sa_decay")
	plt.cla()

	# Best occurs at schedule = exponential


def four_peaks_ga_population_size():

	problem_name = "Four Peaks"


	fitness = mlrose.FourPeaks()
	optim = mlrose.DiscreteOpt(25, fitness, maximize=True)

	######### Compare population sizes for genetic algorithm

	fitness = []

	for i in range(1,100,1):
		best_state, best_fitness, fitness_curve = mlrose.genetic_alg(optim, curve=True, max_attempts=10, pop_size=i)

		fitness.append({"pop_size": i, "fitness": best_fitness})

	plt.plot([x['pop_size'] for x in fitness], [-y['fitness'] for y in fitness])
	plt.title("Four Peaks: Genetic Algorithm")
	plt.xlabel("Population Size")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("4p_ga_pop_size")
	plt.cla()

	# Best occurs around pop_size = 3


def four_peaks_ga_mut_prob():
	######### Compare mutation rates for genetic algorithm

	problem_name = "Four Peaks"

	fitness = mlrose.FourPeaks()
	optim = mlrose.DiscreteOpt(25, fitness, maximize=True)

	mutations = [0.05, 0.1, 0.2, 0.5]
	colors = ['r', 'b', 'g', 'k']

	for i, mutation in enumerate(mutations):
		best_state, best_fitness, fitness_curve = mlrose.genetic_alg(optim, curve=True, max_attempts=10, pop_size=3, mutation_prob=mutation)

		plt.plot(fitness_curve, lw=0.5, c=colors[i], label=str(mutation))

	plt.title("Four Peaks: Genetic Algorithm \n Mutation Probabilities")
	plt.xlabel("No. of Iterations")
	plt.legend(loc="lower right")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("4p_ga_mut_probs")
	plt.cla()

	# Best occurs at mutation_prob = 0.05


def four_peaks_m_pop_size():
	######### Compare population sizes for mimic

	problem_name = "Four Peaks"

	fitness = mlrose.FourPeaks()
	optim = mlrose.DiscreteOpt(25, fitness, maximize=True)

	fitness = []

	for i in range(1,200,2):
		best_state, best_fitness, fitness_curve = mlrose.mimic(optim, curve=True, max_attempts=3, pop_size=i)

		fitness.append({"pop_size": i, "fitness": best_fitness})

	plt.plot([x['pop_size'] for x in fitness], [-y['fitness'] for y in fitness])
	plt.title("Four Peaks: MIMIC")
	plt.xlabel("Population Size")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("4p_mimic_pop_size")
	plt.cla()

	# Best occurs at pop_size = 5


def four_peaks_m_keep_pct():
	######### Compare mutation rates for mimic

	problem_name = "Four Peaks"

	fitness = mlrose.FourPeaks()
	optim = mlrose.DiscreteOpt(25, fitness, maximize=True)

	fitness = []

	for i in range(5, 100, 5):
		best_state, best_fitness, fitness_curve = mlrose.mimic(optim, curve=True, max_attempts=10, pop_size=5, keep_pct=(i / 100))
		fitness.append({'keep_pct': (i / 100), 'fitness': best_fitness})
		
	plt.plot([x['keep_pct'] for x in fitness],[y['fitness'] for y in fitness])
	plt.title("Four Peaks: MIMIC \n Keep Probabilities, Pop size: 5")
	plt.xlabel("Keep Pct")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("4p_mimic_keep_probs")
	plt.cla()

	#### Best occurs at keep_pct = 0.4

def four_peaks_sa_sched_params():

	problem_name = "Four Peaks"

	fitness = mlrose.FourPeaks()
	optim = mlrose.DiscreteOpt(25, fitness, maximize=True)

	######## Compare params for decay schedule 

	fitness = []

	for i in range(10,100,10):

		schedule = mlrose.ExpDecay(init_temp=10, exp_const=0.05, min_temp=(i/10))
		best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(optim, curve=True, max_attempts=10, schedule=schedule)
		fitness.append({"min_temp": (i/10), "fitness": best_fitness})

	plt.plot([x['min_temp'] for x in fitness],[y['fitness'] for y in fitness])
	plt.title("Four Peaks: Simmulated Annealing \n Exp Decay, 100 Attempts")
	plt.xlabel("Min Temp")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("4p_sa_exp_min_temp")
	plt.cla()

	##### Best occurs at min_temp = 3

	fitness = []

	for i in range(5,100,5):

		schedule = mlrose.ExpDecay(init_temp=10, exp_const=(i / 100), min_temp=1)
		best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(optim, curve=True, max_attempts=10, schedule=schedule)
		fitness.append({"exp_const": (i / 100), "fitness": best_fitness})

	plt.plot([x['exp_const'] for x in fitness],[y['fitness'] for y in fitness])
	plt.title("Four Peaks: Simmulated Annealing \n Exp Decay, 10 Attempts")
	plt.xlabel("Exp Constant")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("4p_sa_exp_const")
	plt.cla()

	##### Best occurs at exp_const = 0.55 or so 





def get_weights(n):

	return [np.random.randint(1,11) for i in range(n)]



def get_values(n):

	return [np.random.randint(1,11) for i in range(n)]




def knapsack():

	# from get_weights()
	weights = [10, 6, 3, 7, 6, 4, 2, 6, 2, 9, 6, 3, 7, 8, 4, 5, 5, 7, 5, 8]

	# from get_values()
	values = [3, 6, 10, 8, 10, 2, 10, 6, 2, 7, 3, 2, 2, 5, 3, 9, 2, 5, 5, 5]

	max_weight_pct = 0.5

	problem_name = "Knapsack"

	models = [{"title": "Simulated Annealing", "abbrev": 'sa'}, 
				{"title": "Random Hill Climb", "abbrev": 'rhc'}, 
				{"title": "Genetic Algorithm", "abbrev": 'ga'}, 
				{"title": "MIMIC", "abbrev": 'mimic'}]


	fitness = mlrose.Knapsack(weights, values, max_weight_pct)
	optim = mlrose.DiscreteOpt(len(weights), fitness, maximize=True)

	for model in models: 
		
		fitness_scores = []
		times = []

		sa_attempts = [1, 10, 100, 1000] #1, 10, 100, 1000, 10000, 100000
		m_attempts = [1, 10, 100]
		rhc_restarts = [1, 10, 100, 1000] #1, 10, 100, 1000, 10000, 100000
		ga_attempts = [1, 10, 100, 1000] #1, 10, 100, 1000, 10000, 100000

		colors = ['b', 'r', 'g', 'k']

		if model["abbrev"] == "sa":

			for index, attempt in enumerate(sa_attempts):

				time_start = time.time()
				best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(optim, curve = True, max_attempts = attempt)
				time_end = time.time()
				times.append({"max_attempts": attempt, "duration": round(time_end - time_start, 3)})

				fitness_scores.append(best_fitness)


				plt.plot(fitness_curve, c=colors[index], lw=0.5, label=str(attempt) + " attempts")

			plt.title(problem_name + ": " + str(model['title']))
			plt.xlabel("No. of Iterations")
			plt.ylabel("Fitness")
			plt.legend(loc="lower right")
			plt.xscale("log")
			plt.tight_layout()
			plt.savefig("k_" + str(model["abbrev"]))
			plt.cla()


		if model['abbrev'] == "rhc":

			for index, restart in enumerate(rhc_restarts):

				time_start = time.time()
				best_state, best_fitness, fitness_curve = mlrose.random_hill_climb(optim, curve=True, restarts=restart)
				time_end = time.time()
				times.append({"restarts": restart, "duration": round(time_end - time_start, 3)})

				fitness_scores.append(best_fitness)


				plt.plot(fitness_curve, c=colors[index], lw=0.5, label=str(attempt) + " attempts")

			plt.title(problem_name + ": " + str(model['title']))
			plt.xlabel("No. of Iterations")
			plt.ylabel("Fitness")
			plt.legend(loc="lower right")
			plt.xscale("log")
			plt.tight_layout()
			plt.savefig("k_" + str(model["abbrev"]))
			plt.cla()
				

		if model['abbrev'] == "ga":

			for index, attempt in enumerate(ga_attempts):
				
				time_start = time.time()
				best_state, best_fitness, fitness_curve = mlrose.genetic_alg(optim, pop_size=10, curve=True, max_attempts=attempt)
				time_end = time.time()
				times.append({"max_attempts": attempt, "duration": round(time_end - time_start, 3)})

				fitness_scores.append(best_fitness)


				plt.plot(fitness_curve, c=colors[index], lw=0.5, label=str(attempt) + " attempts")

			plt.title(problem_name + ": " + str(model['title']))
			plt.xlabel("No. of Iterations")
			plt.ylabel("Fitness")
			plt.legend(loc="lower right")
			plt.xscale("log")
			plt.tight_layout()
			plt.savefig("k_" + str(model["abbrev"]))
			plt.cla()


		if model['abbrev'] == 'mimic':

			for index, attempt in enumerate(m_attempts):

				time_start = time.time()
				best_state, best_fitness, fitness_curve = mlrose.mimic(optim, curve=True, max_attempts=attempt)
				time_end = time.time()
				times.append({"max_attempts": attempt, "duration": round(time_end - time_start, 3)})

				fitness_scores.append(best_fitness)



				plt.plot(fitness_curve, c=colors[index], lw=0.5, label=str(attempt) + " attempts")

			plt.title(problem_name + ": " + str(model['title']))
			plt.xlabel("No. of Iterations")
			plt.ylabel("Fitness")
			plt.legend(loc="lower right")
			plt.xscale("log")
			plt.tight_layout()
			plt.savefig("k_" + str(model["abbrev"]))
			plt.cla()

	####### Generate shared duration vs problem size graph
	schedule = mlrose.Arithmetic(init_temp=1.0, decay=0.0001, min_temp=0.001)

	sa_state, sa_fitness, sa_curve = mlrose.simulated_annealing(optim, curve = True, max_attempts=100, schedule=schedule)
	rhc_state, rhc_fitness, rhc_curve = mlrose.random_hill_climb(optim, curve=True, restarts=100)
	ga_state, ga_fitness, ga_curve = mlrose.genetic_alg(optim, pop_size=630, mutation_prob=0.5, curve=True, max_attempts=200)
	m_state, m_fitness, m_curve = mlrose.mimic(optim, curve=True, max_attempts=6, keep_pct=0.2, pop_size=700)
	plt.plot(sa_curve, c="r", lw="0.5", label="Simmulated Annealing")
	plt.plot(ga_curve, c="b", lw="0.5", label="Genetic Alg")
	plt.plot(rhc_curve, c="g", lw="0.5", label="Random Hill Climb")
	plt.plot(m_curve, c="c", lw="0.5", label="MIMIC")
	plt.title("Knapsack \n 10 Attempts")
	plt.xlabel("No. of Iterations")
	plt.ylabel("Fitness")
	plt.xscale("log")
	plt.legend(loc="lower right")
	plt.savefig("k_fitness_combined")
	plt.cla()



def k_sa_decay_schedules(): 

	######## Generate decay schedule 
	schedules = [
		{"title": "Geometric", "schedule": mlrose.GeomDecay(init_temp=10, decay=0.95, min_temp=1), "abbrev": "geom"},
		{"title": "Arithmetic", "schedule": mlrose.ArithDecay(init_temp=10, decay=0.95, min_temp=1), "abbrev": "arith"},
		{"title": "Exponential", "schedule": mlrose.ExpDecay(init_temp=10, exp_const=0.05, min_temp=1), "abbrev": "exp"}]

	weights = [10, 6, 3, 7, 6, 4, 2, 6, 2, 9, 6, 3, 7, 8, 4, 5, 5, 7, 5, 8]
	values = [3, 6, 10, 8, 10, 2, 10, 6, 2, 7, 3, 2, 2, 5, 3, 9, 2, 5, 5, 5]

	max_weight_pct = 0.5

	problem_name = "Knapsack"

	fitness = mlrose.Knapsack(weights, values, max_weight_pct)
	optim = mlrose.DiscreteOpt(len(weights), fitness, maximize=True)

	for index, schedule in enumerate(schedules):
		best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(optim, curve=True, max_attempts=10, schedule=schedule["schedule"])

		plt.plot(fitness_curve, c=colors[index], lw=0.5, label=str(schedule['title']))

	plt.title(problem_name + ": " + str(model['title']) + " \n Decay Schedules, 10 Attempts")
	plt.xlabel("No. of Iterations")
	plt.ylabel("Fitness")
	plt.legend(loc="lower right")
	plt.xscale("log")
	plt.tight_layout()
	plt.savefig("k_sa_decay")
	plt.cla()

def k_sched_param_decay():
	weights = [10, 6, 3, 7, 6, 4, 2, 6, 2, 9, 6, 3, 7, 8, 4, 5, 5, 7, 5, 8]
	values = [3, 6, 10, 8, 10, 2, 10, 6, 2, 7, 3, 2, 2, 5, 3, 9, 2, 5, 5, 5]
	max_weight_pct = 0.5

	fitness_list = []

	problem_name = "Knapsack"

	fitness = mlrose.Knapsack(weights, values, max_weight_pct)
	optim = mlrose.DiscreteOpt(len(weights), fitness, maximize=True)

	for i in range(1,100):
		schedule = mlrose.ArithDecay(init_temp=10, decay=(1/100), min_temp=1)
		best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(optim, curve=True, max_attempts=10, schedule=schedule)

		fitness_list.append([i/100, best_fitness])

	plt.plot([x[0] for x in fitness_list], [y[1] for y in fitness_list])
	plt.title("Knapsack: Simmulated Annealing \n Arithmetic, 10 Attempts")
	plt.xlabel("Temperature Decay")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("k_sa_arith_decay_rate")
	plt.cla()

	###### No single best. Use 0.5 as it is in the middle.

def k_sched_param_min_temp():

	weights = [10, 6, 3, 7, 6, 4, 2, 6, 2, 9, 6, 3, 7, 8, 4, 5, 5, 7, 5, 8]
	values = [3, 6, 10, 8, 10, 2, 10, 6, 2, 7, 3, 2, 2, 5, 3, 9, 2, 5, 5, 5]
	max_weight_pct = 0.5

	fitness_list = []

	problem_name = "Knapsack"

	fitness = mlrose.Knapsack(weights, values, max_weight_pct)
	optim = mlrose.DiscreteOpt(len(weights), fitness, maximize=True)

	for i in range(1,100):
		schedule = mlrose.ArithDecay(init_temp=10, decay=0.5, min_temp=(1/10))
		best_state, best_fitness, fitness_curve = mlrose.simulated_annealing(optim, curve=True, max_attempts=10, schedule=schedule)

		fitness_list.append([i/10, best_fitness])

	plt.plot([x[0] for x in fitness_list],[y[1] for y in fitness_list])
	plt.title("Knapsack: Simmulated Annealing \n Arithmetic, 10 Attempts")
	plt.xlabel("Minimum Temperature")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("k_sa_arith_min_temp")
	plt.cla()


	##### No single best. Use 5 as it is in the middle. 



def k_combined_fitness_chart():
	# Generate combined fitness curve chart
	sa_state, sa_fitness, sa_curve = mlrose.simulated_annealing(optim, curve = True, max_attempts=100)
	rhc_state, rhc_fitness, rhc_curve = mlrose.random_hill_climb(optim, curve=True, restarts=100)
	ga_state, ga_fitness, ga_curve = mlrose.genetic_alg(optim, pop_size=10, curve=True, max_attempts=100)
	m_state, m_fitness, m_curve = mlrose.mimic(optim, curve=True, max_attempts=10)
	plt.plot(sa_curve, c="r", lw="0.5", label="Simmulated Annealing")
	plt.plot(ga_curve, c="b", lw="0.5", label="Genetic Alg")
	plt.plot(rhc_curve, c="g", lw="0.5", label="Random Hill Climb")
	plt.plot(m_curve, c="c", lw="0.5", label="MIMIC")
	plt.title("Knapsack")
	plt.xlabel("No. of Iterations")
	plt.ylabel("Fitness")
	plt.xscale("log")
	plt.legend(loc="lower right")
	plt.savefig("k_fitness_combined")
	plt.cla()		


def k_ga_pop_size():
	######### Compare population sizes for genetic algorithm
	fitness = []

	weights = [10, 6, 3, 7, 6, 4, 2, 6, 2, 9, 6, 3, 7, 8, 4, 5, 5, 7, 5, 8]
	values = [3, 6, 10, 8, 10, 2, 10, 6, 2, 7, 3, 2, 2, 5, 3, 9, 2, 5, 5, 5]

	max_weight_pct = 0.5

	problem_name = "Knapsack"

	fitness = mlrose.Knapsack(weights, values, max_weight_pct)
	optim = mlrose.DiscreteOpt(len(weights), fitness, maximize=True)

	for i in range(100,1100,100):
		best_state, best_fitness, fitness_curve = mlrose.genetic_alg(optim, curve=True, max_attempts=10, pop_size=i)

		fitness.append({"pop_size": i, "fitness": best_fitness})

	plt.plot([x['pop_size'] for x in fitness], [y['fitness'] for y in fitness])
	plt.title("Knapsack: Genetic Algorithm")
	plt.xlabel("Population Size")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("k_ga_pop_size")
	plt.cla()


	##### Best occurs around 600

def k_ga_mut_prob():
	######### Compare mutation rates for genetic algorithm

	weights = [10, 6, 3, 7, 6, 4, 2, 6, 2, 9, 6, 3, 7, 8, 4, 5, 5, 7, 5, 8]
	values = [3, 6, 10, 8, 10, 2, 10, 6, 2, 7, 3, 2, 2, 5, 3, 9, 2, 5, 5, 5]

	max_weight_pct = 0.5

	problem_name = "Knapsack"

	fitness = mlrose.Knapsack(weights, values, max_weight_pct)
	optim = mlrose.DiscreteOpt(len(weights), fitness, maximize=True)

	mutations = [0.05, 0.1, 0.2, 0.5]
	colors = ['r', 'b', 'g', 'k']

	for i, mutation in enumerate(mutations):
		best_state, best_fitness, fitness_curve = mlrose.genetic_alg(optim, curve=True, max_attempts=10, pop_size=500, mutation_prob=mutation)

		plt.plot(fitness_curve, lw=0.5, c=colors[i], label=str(mutation))

	plt.title("Knapsack: Genetic Algorithm \n Mutation Probabilities")
	plt.xlabel("No. of Iterations")
	plt.legend(loc="lower right")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("k_ga_mut_probs")
	plt.cla()


	##### Best occurs around 0.1


def k_m_pop_size():

	weights = [10, 6, 3, 7, 6, 4, 2, 6, 2, 9, 6, 3, 7, 8, 4, 5, 5, 7, 5, 8]
	values = [3, 6, 10, 8, 10, 2, 10, 6, 2, 7, 3, 2, 2, 5, 3, 9, 2, 5, 5, 5]

	max_weight_pct = 0.5

	problem_name = "Knapsack"

	fitness = mlrose.Knapsack(weights, values, max_weight_pct)
	optim = mlrose.DiscreteOpt(len(weights), fitness, maximize=True)

	######### Compare population sizes for mimic

	fitness = []

	for i in range(100,1000,100):
		best_state, best_fitness, fitness_curve = mlrose.mimic(optim, curve=True, max_attempts=3, pop_size=i)

		fitness.append({"pop_size": i, "fitness": best_fitness})

	plt.plot([x['pop_size'] for x in fitness], [y['fitness'] for y in fitness])
	plt.title("Knapsack: MIMIC")
	plt.xlabel("Population Size")
	plt.ylabel("Fitness")
	plt.tight_layout()
	plt.savefig("k_mimic_pop_size")
	plt.cla()


	# Doesn't change. Use 100

def k_m_keep_pct():

	weights = [10, 6, 3, 7, 6, 4, 2, 6, 2, 9, 6, 3, 7, 8, 4, 5, 5, 7, 5, 8]
	values = [3, 6, 10, 8, 10, 2, 10, 6, 2, 7, 3, 2, 2, 5, 3, 9, 2, 5, 5, 5]

	max_weight_pct = 0.5

	problem_name = "Knapsack"

	fitness = mlrose.Knapsack(weights, values, max_weight_pct)
	optim = mlrose.DiscreteOpt(len(weights), fitness, maximize=True)

	######## Compare mutation rates for mimic
	mutations = [0.05, 0.1, 0.2, 0.5]
	colors = ['r', 'b', 'g', 'k']

	for i, mutation in enumerate(mutations):
		best_state, best_fitness, fitness_curve = mlrose.mimic(optim, curve=True, max_attempts=10, pop_size=100, keep_pct=mutation)

		plt.plot(fitness_curve, lw=0.5, c=colors[i], label=str(mutation))

	plt.title("Knapsack: MIMIC \n Keep Probabilities")
	plt.xlabel("No. of Iterations")
	plt.ylabel("Fitness")
	plt.legend(loc="lower right")
	plt.tight_layout()
	plt.savefig("k_mimic_keep_probs")
	plt.cla()


	####### Best occurs at 0.05

def k_problem_size_duration():

	weights_sm = [10, 6, 3, 7, 6, 4, 2, 6, 2, 9, 6, 3, 7, 8, 4, 5, 5, 7, 5, 8]
	values_sm = [3, 6, 10, 8, 10, 2, 10, 6, 2, 7, 3, 2, 2, 5, 3, 9, 2, 5, 5, 5]

	weights_md = [9, 4, 8, 8, 4, 7, 6, 3, 6, 5, 2, 2, 4, 10, 2, 4, 2, 6, 3, 3, 7, 5, 5, 5, 7, 7, 2, 1, 8, 4, 3, 7, 5, 4, 10, 10, 9, 3, 3, 5, 3, 1, 7, 4, 9, 2, 4, 5, 1, 9]
	values_md = [7, 7, 5, 8, 8, 4, 2, 5, 4, 9, 10, 1, 2, 6, 2, 10, 1, 4, 7, 5, 1, 8, 7, 6, 7, 1, 8, 1, 5, 8, 6, 1, 1, 4, 1, 9, 4, 1, 5, 7, 9, 5, 6, 5, 1, 7, 5, 9, 10, 4]

	weights_lg = [8, 10, 10, 1, 6, 5, 2, 10, 8, 1, 2, 1, 7, 1, 7, 4, 9, 10, 2, 10, 3, 4, 3, 8, 2, 4, 2, 8, 3, 2, 10, 9, 9, 9, 3, 8, 1, 10, 5, 6, 7, 4, 8, 7, 7, 5, 8, 8, 3, 10, 10, 3, 3, 
	8, 8, 2, 4, 7, 6, 2, 2, 5, 5, 6, 3, 5, 1, 4, 10, 2, 2, 3, 3, 8, 3, 2, 2, 10, 3, 2, 5, 8, 6, 1, 6, 6, 4, 9, 7, 9, 9, 3, 5, 3, 10, 8, 5, 3, 3, 10]
	values_lg = [9, 7, 8, 9, 1, 2, 3, 5, 5, 1, 4, 1, 5, 7, 9, 7, 9, 7, 1, 4, 7, 7, 1, 10, 6, 7, 3, 6, 3, 9, 9, 6, 10, 6, 10, 3, 2, 5, 10, 3, 5, 5, 10, 10, 3, 1, 9, 6, 3, 9, 4, 8, 8, 3, 8, 
	9, 9, 1, 7, 8, 1, 9, 7, 3, 7, 7, 4, 6, 5, 1, 1, 2, 8, 10, 4, 8, 6, 4, 10, 5, 6, 9, 3, 3, 9, 4, 5, 10, 2, 4, 4, 6, 1, 3, 6, 5, 10, 5, 1, 1]

	all_weights = [weights_sm, weights_md, weights_lg]
	all_values = [values_sm, values_md, values_lg]

	max_weight_pct = 0.5

	sizes = [len(x) for x in all_weights]

	markers = ['o', 'x', '^']
	legend_items = [mlines.Line2D([], [], color='k', marker='o', linestyle='None', markersize=5, label='n 20'),
		mlines.Line2D([], [], color='k', marker='x', linestyle='None', markersize=5, label='n 50'),
		mlines.Line2D([], [], color='k', marker='^', linestyle='None', markersize=5, label='n 100'),
		mlines.Line2D([0], [0], color='g', lw=2),
		mlines.Line2D([0], [0], color='b', lw=2),
		mlines.Line2D([0], [0], color='r', lw=2),
		mlines.Line2D([0], [0], color='c', lw=2)]

	legend_labels = ['n 20', 'n 50', 'n 100', 'SA', 'RHC', 'GA', 'MIMIC']

	for index, weight in enumerate(all_weights):

		fitness = mlrose.Knapsack(weight, all_values[index], max_weight_pct)
		optim = mlrose.DiscreteOpt(len(weight), fitness, maximize=True)

		schedule = mlrose.ExpDecay(init_temp=10, exp_const=0.55, min_temp=3)
		
		start = time.time()
		sa_state, sa_fitness, sa_curve = mlrose.simulated_annealing(optim, curve=True, max_attempts=10, schedule=schedule)
		duration = time.time() - start

		plt.plot(duration,sa_fitness, lw=0.5, color='g', marker=markers[index])


	for index, weight in enumerate(all_weights):

		fitness = mlrose.Knapsack(weight, all_values[index], max_weight_pct)
		optim = mlrose.DiscreteOpt(len(weight), fitness, maximize=True)
		
		start = time.time()
		rhc_state, rhc_fitness, rhc_curve = mlrose.random_hill_climb(optim, curve=True, restarts=10)
		duration = time.time() - start

		plt.plot(duration,rhc_fitness, lw=0.5, color='b', marker=markers[index])


	for index, weight in enumerate(all_weights):

		fitness = mlrose.Knapsack(weight, all_values[index], max_weight_pct)
		optim = mlrose.DiscreteOpt(len(weight), fitness, maximize=True)
		
		start = time.time()
		ga_state, ga_fitness, ga_curve = mlrose.genetic_alg(optim, curve=True, max_attempts=10, pop_size=600, mutation_prob=0.1)		
		duration = time.time() - start

		plt.plot(duration,ga_fitness, lw=0.5, color='r', marker=markers[index])

	for index, weight in enumerate(all_weights):

		fitness = mlrose.Knapsack(weight, all_values[index], max_weight_pct)
		optim = mlrose.DiscreteOpt(len(weight), fitness, maximize=True)
		
		start = time.time()
		m_state, m_fitness, m_curve = mlrose.mimic(optim, curve=True, max_attempts=10, pop_size=100, keep_pct=0.05)		
		duration = time.time() - start

		plt.plot(duration,m_fitness, lw=0.5, color='c', marker=markers[index])


	plt.title("Knapsack \n Problem Size")
	plt.xlabel("Duration")
	plt.xscale("log")
	plt.ylabel("Fitness")
	plt.legend(legend_items, legend_labels, loc="lower right")
	plt.savefig("k_duration_vs_fitness")
	plt.cla()

		

if __name__ == '__main__':
	
	# print(get_salesman_coords(25))
	# salesman()

	# get_6_peaks_state(10)
	# four_peaks()
	# four_peaks_problem_size_duration()

	# print(get_values(50))
	# print(get_weights(50))


	# print(get_values(100))
	# print(get_weights(100))
	# knapsack()
	# k_problem_size_duration()
	# k_sched_param_min_temp()


		
