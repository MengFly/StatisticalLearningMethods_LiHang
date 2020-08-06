import matplotlib.pyplot as plt
import numpy as np

e_m = np.arange(0.01, 0.99, 0.01)


def a_m(em):
    return 0.5 * np.log((1 - em) / em)


plt.plot(e_m, a_m(e_m))
plt.show()
