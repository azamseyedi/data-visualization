import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

X, y = make_blobs(
    200,
    n_features=3,
    centers=3,
    cluster_std=2,
    random_state=37
)

fig = plt.figure()
ax = plt.axes(projection="3d")




colors = ["blue", "green", "purple"]

for k, col in enumerate(colors):
    cluster_data = y == k

    ax.scatter3D(
    X[cluster_data, 0],
    X[cluster_data, 1],
    X[cluster_data, 2],
    c=col,
    label=col,
    s=100
)


ax.legend()
plt.show()