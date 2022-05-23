from pandas import *
from numpy import *


class LogisticRegression():
    def __init__(self, lr=0.001, iters=100):
        self.th = None
        self.lr = lr
        self.iters = iters

    def sig(self, t):
        return 1 / (1 + exp(-t))

    def cost(self, x, y, theta):
        h = self.sig(dot(x, theta.T))
        first_eq = multiply(y, -log(h))  # equal to y * -log(h)
        second_eq = multiply(1 - y, -log(1 - h))
        eq = sum(first_eq + second_eq) / x.shape[0]
        return eq

    def fit(self, x, y):
        x = matrix(x)
        y = matrix(y)
        m, n = x.shape
        cost_list = []
        th_list = []
        self.th = matrix(zeros(n))
        j = self.cost(x, y, self.th)
        cost_list.append(j)
        th_list.append(self.th)
        for _ in range(self.iters):
            old_j = j
            old_th = self.th
            h = self.sig(dot(x, self.th.T))
            self.th = self.th - ((self.lr / m) * dot(x.T, (h - y)))
            j = self.cost(x, y, self.th)
            cost_list.append(j)
            th_list.append(self.th)
            if isnan(j) or j > old_j:
                j = old_j
                self.th = old_th
                break

        # print(cost_list)
        # print(th_list)
        # print(j)
        # print(self.th)

    def predict(self, x):
        y_pred = self.sig(dot(x, self.th.T))
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        return y_pred


path = "data.csv"
data = read_csv(path, header=None, names=["Exam1", "Exam2", "Admis"])

# add col to x
data.insert(0, "Ones", 1)
# seperate training data and target data
cols = data.shape[1]
x = data.iloc[:, :cols - 1]
y = data.iloc[:, cols - 1: cols]

x = matrix(x)
y = matrix(y)

x_test = [1, 2, 3, 4, 5, 6.54, 7, 8, 9]
x_test = array(x_test).reshape((3, 3))


# tests
model = LogisticRegression()
model.fit(x, y)
y_pred = model.predict(x_test)
print(y_pred)
# model.cost(x, y, theta)
