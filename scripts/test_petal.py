import numpy as np
import matplotlib.pyplot as plt

W = 500
H = 500

XLIM = 5
YLIM = 5

# length of petal
L = 1

x, y = np.meshgrid(np.linspace(-XLIM, XLIM, W), np.linspace(-YLIM, YLIM, H))
r = np.hypot(x, y)
theta = np.arctan2(y, x)

c = np.logical_and(
  np.logical_and(r < 0.5*L + 0.5*L*np.cos(4*theta), theta > np.pi/4),
  theta < 3*np.pi/4)
im = np.reshape(c, (H, W))

plt.imshow(c, origin='lower', extent=(-XLIM, XLIM, -YLIM, YLIM))
plt.show()