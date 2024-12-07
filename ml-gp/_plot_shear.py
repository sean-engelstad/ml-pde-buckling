
import time
import matplotlib.pyplot as plt
import niceplots
import numpy as np

plt.style.use(niceplots.get_style())

# case 1 with 1x1 panel
# newton = [_ for _ in range(5)]
# loss = [9.934e8, -1.99e6, -6.82e5, -1.312e6, 1.417e3]
# loss = [abs(_) for _ in loss]
# lam = [632.665, 632.665, 654.237, 473.983, 459.930]

# case 2 with 5x1 panel
newton = [_ for _ in range(8)]
loss = [9.16985222e+12, 3.33524349e+08, 4.6185637e+08, 4.66529213e+08, 4.62328198e+08, 4.6236696e+08, 4.62497504e+08, 4.62500952e+08]
lam = [632.6658800064105, 632.6658800064105, 531.6636135680965, 536.452851205531, 539.1594179889079, 539.1034426737846, 539.0176727892925, 539.0165460301461]

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# plt.style.use(niceplots.get_style())
plt.margins(x=0.05, y=0.05)
fig, ax = plt.subplots(2)
ax[0].plot(newton, loss, 'o-')
ax[1].plot(newton, lam, 'o-', color=colors[1])

colors = plt.rcParams
plt.xlabel("Newton iterations")
ax[0].set_ylabel("Loss")
ax[1].set_ylabel("Eigenvalue")
plt.xscale('log')
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].margins(x=0.05, y=0.05)
ax[1].margins(x=0.05, y=0.05)

# plt.show()
plt.savefig('shear-hist.png', dpi=400)
