import numpy as np
import matplotlib.pyplot as plt


data_in = [83.33333333,
76.92307692,
76.92307692,
76.92307692,
76.92307692,
78.57142857,
81.25,
81.25,
82.35294118,
82.35294118,
82.35294118,
85,
86.36363636,
86.36363636,
86.36363636,
86.36363636,
86.36363636,
82.60869565,
84,
85.71428571,
85.71428571,
85.71428571,
85.71428571,
86.20689655,
82.35294118,
82.85714286,
82.85714286,
82.85714286,
82.85714286,
80.55555556,
82.5,
82.92682927,
83.33333333,
83.33333333,
83.33333333,
84.09090909
]

data_in_2 = [75,
83.33333333,
83.33333333,
83.33333333,
83.33333333,
85.71428571,
80,
84.61538462,
84.61538462,
84.61538462,
84.61538462,
86.66666667,
81.25,
76.47058824,
76.47058824,
76.47058824,
76.47058824,
77.77777778,
76.19047619,
76.19047619,
72.72727273,
72.72727273,
72.72727273,
76,
74.07407407,
75,
72.4137931,
72.4137931,
72.4137931,
68.75,
71.42857143,
72.22222222,
72.22222222,
72.22222222,
72.22222222,
72.97297297,
69.23076923,
65.85365854,
65.85365854,
65.85365854,
65.85365854,
64.44444444
]



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



