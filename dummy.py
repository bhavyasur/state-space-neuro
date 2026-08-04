import numpy as np
import matplotlib.pyplot as plt

# Grid
x = np.linspace(-3, 3, 25)
y = np.linspace(-3, 3, 25)
X, Y = np.meshgrid(x, y)

# Dynamics
# Fast attraction toward y=0
# Slow drift along the attractor that vanishes near x=0
U = 0.35 * np.tanh(X)
V = -1.2 * Y

fig, ax = plt.subplots(figsize=(6,6))

ax.quiver(
    X, Y, U, V,
    color='k',
    angles='xy',
    scale_units='xy',
    scale=15,
    width=0.004,
    pivot='mid'
)

ax.set_xlim(-3,3)
ax.set_ylim(-3,3)
ax.set_aspect('equal')

ax.set_xlabel("Latent dimension 1", fontsize=16)
ax.set_ylabel("Latent dimension 2", fontsize=16)

plt.tight_layout()
plt.show()
fig.savefig("lineattractor.svg", format='svg')

