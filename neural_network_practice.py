from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.10,
    random_state=1
)

model = MLPClassifier(
    random_state=1,
    max_iter=5000,
    verbose=True,
    hidden_layer_sizes=(10,)
)

model.fit(X_train, y_train)



WIDTH = 1920
HEIGHT = 1080
LINE_WIDTH = 5

layers = []

for current in model.coefs_:
    layers.append(len(current))

print(model.coefs_[0].shape)
print(model.coefs_[1].shape)

layers.append(model.n_outputs_)

print(layers)



coordinates = []

x_increment = WIDTH / (len(layers) + 1)

x = x_increment


for layer in layers:
    y_increment = HEIGHT / (layer + 1)
    y = y_increment
    layer_coordinates = []

    for node in range(layer):
        layer_coordinates.append((int(x), int(y)))
        y += y_increment

    coordinates.append(layer_coordinates)
    x += x_increment


import cv2
import numpy as np

img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
img.fill(255)


for i in range(0, len(coordinates)):
    for j in range(0, len(coordinates[i])):

        if i < len(coordinates) - 1:
            for k in range(0, len(coordinates[i + 1])):

                current_line_width = int(
                    LINE_WIDTH * (abs(model.coefs_[i][j][k]) / 1)
                )

                if current_line_width == 0:
                    current_line_width = 1

                color = (0, 0, 255)

                if model.coefs_[i][j][k] < 0:
                    color = (255, 0, 0)

                cv2.line(
                    img,
                    coordinates[i][j],
                    coordinates[i + 1][k],
                    color,
                    current_line_width
                )

        cv2.circle(
            img,
            coordinates[i][j],
            40,
            (255, 255, 255),
            -1
        )

        cv2.circle(
            img,
            coordinates[i][j],
            40,
            (0, 0, 0),
            1
        )

        cv2.putText(
            img,
            str(i + 1) + str(j + 1),
            (coordinates[i][j][0] - 18, coordinates[i][j][1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            1,
            cv2.LINE_AA
        )



cv2.namedWindow("The image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("The image", 1000, 600)

cv2.imshow("The image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()