import numpy as np
import matplotlib.pyplot as plt
import random

HS = 0
HGP = 0.002
N = int(1E+5)
Bt = [1 if random.random() >= 0.15 else 0 for i in range(N)]

#simple implementation of the Miller model of habit learning
#bt is 1, action performed. Linearly increase

class Habit:
    def __init__(self, HS, init):
        self.HS = HS
        self.HGP = init["HGP"]

    def habit_update(self, b):
        self.HS = self.HS + self.HGP*(b-self.HS)
        return self.HS


rates = [0.2, 0.05, 0.01, 0.005, 0.001, 0.0005]
current_habit_strength = np.zeros((N, len(rates)))
Btx = np.arange(0,N, 1)
for r in rates:
    habit = Habit(HS, {"HGP": r})
    for n in range(len(Bt)):
        current_habit_strength[n,rates.index(r)] = habit.habit_update(Bt[n])
    plt.plot(Btx, current_habit_strength[:, rates.index(r)], label="rate-{}".format(r))

plt.title("Miller Habit Model")
plt.xlabel("Time step")
plt.ylabel("Habit Str")
plt.legend()
plt.show()