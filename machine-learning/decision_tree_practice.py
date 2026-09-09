# Import train_test_split to divide the dataset into training and testing sets
from sklearn.model_selection import train_test_split

# Import DecisionTreeClassifier to create a decision tree classification model
from sklearn.tree import DecisionTreeClassifier

# Import the Iris dataset, which contains measurements of different Iris flower species
from sklearn.datasets import load_iris

# Import the tree module to access tools such as export_graphviz for visualizing the decision tree
from sklearn import tree

from sklearn.metrics import accuracy_score


import pydotplus
import os



os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"


# Load the Iris dataset and store it in the variable iris
iris = load_iris()


# Split the Iris data into training and testing sets
# iris.data contains the input features (X), such as sepal and petal measurements
# iris.target contains the correct class labels (y): 0, 1, or 2 for the three Iris species
# test_size=0.10 means 10% of the dataset is used for testing and 90% for training
# random_state=1 makes the split reproducible, so we get the same split each time we run the code
X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.10,
    random_state=1
)


# Create a Decision Tree classification model
# random_state=1 makes the behavior of the model reproducible
model = DecisionTreeClassifier(random_state=1)


# Train the Decision Tree using the training features (X_train)
# and their corresponding correct labels (y_train)
model.fit(X_train, y_train)


# Define readable names for the four input features in the Iris dataset
# These names will appear in the visualization of the decision tree
feature_names = [
    "sepal length",
    "sepal width",
    "petal length",
    "petal width"
]


# Convert the trained Decision Tree into Graphviz DOT format
# The result is stored as text in the dot_data variable
dot_data = tree.export_graphviz(

    # Specify the trained Decision Tree that we want to visualize
    model,
    # Use readable feature names instead of feature numbers
    feature_names=feature_names,
    # Use the actual Iris species names for the predicted classes
    # The classes are: setosa, versicolor, and virginica
    class_names=iris.target_names,
    # Add colors to the nodes based on the predicted class
    filled=True,
    # Display the decision tree nodes with rounded corners
    rounded=True
)


y_pred = model.predict(X_test)

print(y_pred)
print(y_test)




accuracy = accuracy_score(y_test, y_pred)

print(accuracy)


graph = pydotplus.graph_from_dot_data(dot_data)

graph.dpi = 300

graph.write_png("ClassificationTree.png")



os.startfile("ClassificationTree.png")

