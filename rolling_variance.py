
import time
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure(figsize=(16,8))
axes = fig.add_subplot(111)
data_plot=plt.plot(0,0)
line, = axes.plot([],[])

tik = time.time()
MAX_ARRAY_SIZE = 64

M = []
S = []
V = []

while(time.time() - tik < 20):
    x = time.time()-tik
    y = np.random.random()

    k = len(M) if len(M) < MAX_ARRAY_SIZE else MAX_ARRAY_SIZE

    Mc = M[k-1] + (y-M[k-1])/k if len(M) != 0 else y
    if(len(M) == MAX_ARRAY_SIZE):
        M = M[1:len(M)]+[Mc]
    else:
        M.append(Mc)

    if(len(S) == 0):
        S.append(0)
        V.append(0)
    elif(len(S) == MAX_ARRAY_SIZE):
        S = S[1:len(S)] + [(S[k-1] + (y - M[k-1])*(y - M[k]))]
        V = V[1:len(V)] + [(S[k-1] + (y - M[k-1])*(y - M[k]))/(k-1)]
    else:
        S.append(S[k-1] + (y - M[k-1])*(y - M[k-1]))
        if(k >= 2):
            V.append((S[k-1] + (y - M[k-1])*(y - M[k]))/(k-1))
        else:
            V.append(S[k-1] + (y - M[k-1])*(y - M[k]))

    print(V)

    line.set_ydata(
        np.append(line.get_ydata(), Mc) if len(line.get_ydata()) != 0 else [Mc]
    )
    line.set_xdata(
        np.append(line.get_xdata(), x) if len(line.get_xdata()) > 0 else [x]
    )

    axes.set_ylim(0, 1) # +1 to avoid singular transformation warning
    axes.set_xlim(0, time.time() - tik)
    plt.draw()
    plt.pause(0.001)

plt.show(block=True)
tik = time.time()

#x = 0
"""while(True):
    if(time.time() - tik > 10*1000):
        break
    tick = np.random.random()
    update(h1, [x, tick])"""




