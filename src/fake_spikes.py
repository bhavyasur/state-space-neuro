"""file to generate fake spikes (poisson)with known dynamics to test preliminary models on 
before applying to real data. using an inhomogenous poisson process"""

"""Assisted by 
https://medium.com/@baxterbarlow/poisson-spike-generators-stochastic-theory-to-python-code-a76f8cc7cc32"""

import numpy as np
import matplotlib.pyplot as plt

def generate_spikes(r, T, dt=0.001):
    """generates inhomogenous poisson spike train with time-dependent firing rate"""

    spike_times = []
    t = 0

    while t < T:
        rate = r(t)
        if rate * dt > np.random.rand():
            spike_times.append(t)
        t += dt

    return np.array(spike_times)

def r(t):
    return 20 + 10 * np.sin(2 * np.pi * t / 1) # example time-varying firing rate (20 Hz baseline with 10 Hz sinusoidal modulation) 


# test
T = 10
spikes = generate_spikes(r, T)

plt.eventplot(spikes, orientation="horizontal", colors='black')
plt.xlabel("Time (s)")
plt.ylabel("Spike")
plt.title("Inhomogeneous Poisson Spike Train")
plt.show()