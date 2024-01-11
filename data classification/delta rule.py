import matplotlib.pyplot as plt
import numpy as np


class Delta_Rule():
    def __init__(self):
        # data initialization
        self.__data = np.array([
            # Class 1
            [4, 1, 1, 1],
            [5, 3, 1, 1],
            [6, -1, 1, 1],
            [7, 4, 1, 1],

            # Class 2
            [1, -5, 1, -1],
            [2, -3, 1, -1],
            [3, -2, 1, -1],
            [4, -3, 1, -1],
            [5, -5, 1, -1],

            # Class 3
            [-1, 6, -1, 1],
            [0, 3, -1, 1],
            [2, 2, -1, 1],
            [3, 3, -1 ,1],
            [4, 5, -1, 1],

            # Class 4
            [-2, 4, -1, -1],
            [-2, -1, -1, -1],
            [-1, 1, -1, -1],
            [-1, -4, -1, -1]]
        )
        
        # weight initialization
        self.__weights_bar = [[0, 0, 0], [0, 0, 0]]

        # learning rate
        self.__learning_rate = 0.01

        # epoch
        self.__num_epochs = 1000


    def make_separating_line(self):
       for epoch in range(self.__num_epochs):
            for row in self.__data:
                x_bar = np.append(1, row[:2])   # input vector with bias (1, x1, x2)
                target = np.array(row[2:])      # target vector (t1, t2)

                y = [np.dot(self.__weights_bar[0], x_bar), np.dot(self.__weights_bar[1], x_bar)]
                
                # weights update
                for i in range(2):
                    if y[i] != target[i]:
                        dw = self.__learning_rate * (x_bar * (target[i] - y[i]))
                        self.__weights_bar[i] += dw
                
            # termination condition : If the largest change in the weights and biases is smaller than a preset tolerance, then stop.
            if abs(y[0] - target[0]) < 1 and abs(y[0] - target[0]) < 1:
                print(f"epoch : {epoch}")
                break


    def data_visualization(self):
        # separating line: (w1 * x) + (w2 * y) + bias = 0
        bias = [self.__weights_bar[0][0], self.__weights_bar[1][0]]
        weights = [self.__weights_bar[0][1:], self.__weights_bar[1][1:]]

        x = np.linspace(-8, 8)

        y1 = - (weights[0][0] * x + bias[0]) / weights[0][1]
        y2 = - (weights[1][0] * x + bias[1]) / weights[1][1]


        # data visualization
        t1 = self.__data[:, 2] # target[0]
        t2 = self.__data[:, 3] # target[1]

        plt.scatter(self.__data[(t1 == 1) & (t2 == 1), 0], self.__data[(t1 == 1) & (t2 == 1), 1], c='b', marker='s', label='Class 1')
        plt.scatter(self.__data[(t1 == 1) & (t2 == -1), 0], self.__data[(t1 == 1) & (t2 == -1), 1], c='r', marker='o', label='Class 2')
        plt.scatter(self.__data[(t1 == -1) & (t2 == 1), 0], self.__data[(t1 == -1) & (t2 == 1), 1], c='g', marker='*', label='Class 3')
        plt.scatter(self.__data[(t1 == -1) & (t2 == -1), 0], self.__data[(t1 == -1) & (t2 == -1), 1], c='y', marker='^', label='Class 4')

        plt.plot(x, y1, 'r--', label='Separating Line1')
        plt.plot(x, y2, 'b--', label='Separating Line2')

        plt.title("Classification by Delta Rule")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.axhline(0, color='black',linewidth=1)
        plt.axvline(0, color='black',linewidth=1)
        plt.xlim(-8, 8)
        plt.ylim(-8, 8)
        plt.legend(loc='upper left')
        plt.grid(True)
        plt.show()


def main():
    print("Classification by Delta Rule")
    p = Delta_Rule()
    p.make_separating_line()
    p.data_visualization()

if __name__ == "__main__":
    main()