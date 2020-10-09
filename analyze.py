import matplotlib.pyplot as plt

import main


def print_salesman():
	coords = main.salesman(50)
	plt.scatter([coord[0] for coord in coords],[coord[1] for coord in coords])
	plt.show()


if __name__ == '__main__':
	print_salesman()