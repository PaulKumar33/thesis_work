import numpy as np
import matplotlib.pyplot as plt

data_in = []
data_in_2 = []

X1 = np.linspace(0, len(data_in)-1, len(data_in))
X12 = np.linspace(0, len(data_in_2)-1, len(data_in_2))

X = np.ones((len(data_in), 2))
for i in range(len(data_in)):
    X[i, 1] = X1[i]

X2 = np.ones((len(data_in_2), 2))
for i in range(len(data_in_2)):
    X2[i, 1] = X12[i]

Y = data_in
Y2 = data_in_2


def linear_fit(X, Y):
    inner = np.linalg.inv(np.matmul(np.transpose(X), X))
    m1 = np.matmul(inner, np.transpose(X))
    beta = np.matmul(m1, Y)

    return beta

def movavg(x, order):
    _x = np.array([])
    for i in range(len(x)):
        if(i < order-1):
            temp = sum(x[0:i+1])/(i+1)
            _x = np.concatenate((_x, [temp]), axis=None)
        else:
            _x = np.concatenate((_x, [sum(x[i:i + order])/(order)]), axis=None)
    return _x


beta = linear_fit(X, Y)
beta_2 = linear_fit(X2, Y2)
Y_new = np.matmul(X, beta)
Y_new2 = np.matmul(X2, beta_2)

plt.scatter(X1, data_in, label="Raw Indicator Data")
plt.scatter(X12, data_in_2, label="Raw Data")

plt.plot(X1, Y_new, 'g-', label="{0}X+{1}".format(beta[1], beta[0]))
plt.plot(X2[:,1], Y_new2, 'r-', label="{0}X+{1}".format(beta_2[1], beta_2[0]))
plt.ylim(50,100)
plt.legend()
plt.grid()
plt.title("HW Compliance comparison between Indicators and No indicators")
plt.xlabel("Sample Point")
plt.ylabel("Compliance Rate")
plt.show()
