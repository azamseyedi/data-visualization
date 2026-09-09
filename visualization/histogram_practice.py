import matplotlib.pyplot as plt

data = [10, 12, 12, 14, 15, 15, 15, 18, 20, 21, 22, 22, 25]

plt.hist(data)

plt.show()

import matplotlib.pyplot as plt
import numpy as np

data = np.random.randint(0, 500, 100)

fig, ax = plt.subplots()

ax.hist(
    data,
    bins=10,
    linewidth=1,
    color="lightgray",
    edgecolor="blue"
)

plt.show()

fig.savefig("Histogram.jpg")