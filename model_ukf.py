import numpy as np


class ModelState:

    def __init__(self, n, x_0):

        self.n = n
        self.x = np.zeros((n,2))
        self.y = np.zeros((n,2))

        self.xi = np.zeros(2)
        self.yi = np.zeros(2)

        self.x[0, :] = x_0[:, 0]

        self.u = np.random.normal(0., 1.0, (n, 2))  # process noise
        self.v = np.random.normal(0., 1.0, (n, 2))  # measurement noise

    def generate_data(self):
        for i in range(1, self.n):
            self.x[i, :] = self.model_state(i, self.x[i-1, :]) + self.u[i, :]
            self.y[i, :] = self.model_output(self.x[i, :]) + self.v[i, :]

    def model_state(self, i, x_p):
        self.xi[0] = 0.5 * x_p[0] - 0.1 * x_p[1] + 0.7 * (x_p[0]/(1 + x_p[0]**2)) + 2.2 * np.cos(1.2 * (i-1))
        self.xi[1] = 0.8 * x_p[1] - 0.2 * x_p[0] + 0.9 * (x_p[1] / (1 + x_p[1] ** 2)) + 2.4 * np.cos(2.2 * (i - 1))
        return self.xi

    def model_output(self, x_i):
        self.yi[0] = (x_i[0] * 2) / 9.0 + (x_i[1] * 2) / 7.0
        self.yi[1] = (x_i[0] * 2) / 6.0 + (x_i[1] * 2) / 2.0
        return self.yi
