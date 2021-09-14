import numpy as np
import matplotlib.pyplot as plt
import random

HS = 0
HGP = 0.002
N = int(1E+3)
Bt = [1 if random.random() >= 0.15 else 0 for i in range(N)]

#simple implementation of the Miller model of habit learning
#bt is 1, action performed. Linearly increase
class BEHAVIOUR_EVENT_GENERATION:
    def __init__(self):
        pass

    def defined_event(self, e=0, length=1E+3, prob=None, ending=None):
        '''
        creates a defined event
        0 - cnt event
        1 - random
        2 - given some probability
        3 - end at some point
        '''

        if(e==0):
            return [1 for i in range(length)]
        elif(e==1):
            return [random.randint(0,1) for i in range(length)]
        elif(e==2):
            p = []
            for i in range(length):
                p.append(1 if random.random() <= p else 0)
            return p
        elif(e == 3):
            if(ending <= 1):
                ending = length*ending
            return [1 if i < ending else 0 for i in range(length)]


class Habit:
    def __init__(self, HS, init):
        self.HS = HS
        self.HGP = init["HGP"]

    def habit_update(self, b):
        self.HS = self.HS + self.HGP*(b-self.HS)
        return self.HS

class KLEIN_HEBBIAN:
    '''simulates the KLEIN model of habitual learning'''
    def __init__(self, nu, zeta):
        self.nu = nu
        self.zeta = zeta
        self.w1 = 0

    def run_habit_simulation(self, steps, type="cnt"):
        '''runs the habit simualtion'''
        types = dict(cnt=0, random=1, prob=2, ending=3)
        nu_iterator = [self.nu] if not isinstance(self.nu, list) else self.nu
        zeta_iterator = [self.zeta] if not isinstance(self.zeta, list) else self.zeta

        Events = BEHAVIOUR_EVENT_GENERATION()
        if(type=="ending"):
            events = Events.defined_event(e=types[type], length=steps, ending=0.5)
        else:
            events = Events.defined_event(length=steps)
        #blah blah blah complexity
        measured_habits = {}
        for n in nu_iterator:
            for z in zeta_iterator:
                if(f"{n}_{z}" not in measured_habits.keys()):
                    measured_habits[f"{n}_{z}"] = []
                for i in range(steps):
                    w2 = self.w1 + (n*events[i]*(1-self.w1) - z*self.w1)
                    self.w1 = w2
                    measured_habits[f"{n}_{z}"].append(w2)

        return measured_habits



'''rates = list(np.arange(0.01, 0.05, 0.01))
current_habit_strength = np.zeros((N, len(rates)))
Btx = np.arange(0,N, 1)
for r in rates:
    habit = Habit(HS, {"HGP": r})
    for n in range(len(Bt)):
        current_habit_strength[n,rates.index(r)] = habit.habit_update(Bt[n])
    plt.plot(Btx, current_habit_strength[:, rates.index(r)], label="rate-{:.2f}".format(r))

plt.title("Miller Habit Model")
plt.xlabel("Time step")
plt.ylabel("Habit Str")
plt.grid()
plt.legend()
plt.show()'''
nu = 0.01
zeta = 0.01
klein = KLEIN_HEBBIAN(nu, zeta)
s = 1000
sim = klein.run_habit_simulation(s, type="ending")


import plotly.graph_objects as go
fig = go.Figure([go.Scatter(
    x = [i+1 for i in range(s)],
    y = sim[f"{nu}_{zeta}"],
    line=dict(color="rgb(0,0,255)"),
    mode='lines'
)])

fig.show()