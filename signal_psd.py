import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import time

import mcp_3008_driver as mcp

mcp = mcp.mcp_external()

x = []

print("Collecting")
tik = time.time()
while(time.time() - tik < 10):
    x.append(mcp.read_IO(0)/65355*5)
    
#done collection
Ts = 10-tik
Fs = len(x)/Ts

plt.plot(x)
plt.xlabel("Sample")
plt.ylabel("Voltage")
plt.show()
x = np.array(x)
#f, Pxx = signal.welch(x, Fs)
plt.psd(x**2, 1024)

plt.xlabel('frequency [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.show()
    
