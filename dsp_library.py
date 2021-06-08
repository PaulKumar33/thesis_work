import matplotlib.pyplot as plt
import numpy as np
#from scipy import signal
import time

data1 = [100,
75,
85.71428571,
50,
75,
83.33333333,
62.5,
20,
50]

data2 = [100,
100,
100,
66.66666667,
85.71428571,
100,
85.36585366
]


class DSP(object):

    def __init(self):
        pass

    def MA_Model(self):
        pass

    def MA_Filter(self, data_in, order):
        ma_calc = []
        data_in = np.array(data_in)

        #calculate the initial values
        for i in range(1, order):
            temp = sum(data_in[0:i])
            ma_calc.append(temp/len(data_in[0:i]))

        #now for the remaining data pts
        for i in range(order-1, len(data_in)):
            print(i-order, i)
            temp = sum(data_in[i-order+1:i+1])
            ma_calc.append(temp/order)

        return ma_calc

if __name__ == "__main__":
    dsp = DSP()
    d1 = dsp.MA_Filter(data1, 3)
    print(d1)
    x1 = np.linspace(1,len(data1), len(data1))
    x11 = np.linspace(1, len(data1), len(data1))
    x_labels = ["April-4th", "April-5th", "April-6th", "April-7th", "April-8th", "April-9th", "April-10th", "April-11th", "April-12th"]

    plt.plot(x1, data1, label="Raw measurements")
    plt.plot(x1, d1, label="3-pt Moving Average")
    plt.xticks(x1, labels=x_labels)
    plt.ylabel("Compliance [%]")
    plt.legend()
    plt.title("Hand Washing Compliance With No Reminders")
    plt.grid()

    plt.figure(2)
    x_labels = ["April-12th", "April-13th", "April-14th", "April-15th", "April-16th", "April-17th", "April-18th"]
    x2 = np.linspace(1, len(data2), len(data2))
    d2 = dsp.MA_Filter(data2, 3)
    plt.plot(x2, data2, label="Raw measurements")
    plt.plot(x2, d2, label="3-pt Moving average")
    plt.xticks(x2, labels=x_labels)
    plt.title("Hand Washing Compliance With Reminders")
    plt.ylabel("Compliance [%]")
    plt.legend()
    plt.grid()

    plt.show()
