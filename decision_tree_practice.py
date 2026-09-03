from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

from sklearn import tree

iris = load_iris()


X_train, X_test, y_train, y_test = train_test_split(
    iris.data,
    iris.target,
    test_size=0.10,
    random_state=1
)


model = DecisionTreeClassifier(random_state=1)
model.fit(X_train, y_train)


feature_names = ["sepal length", "sepal width", "petal length", "petal width"]

dot_data = tree.export_graphviz(
    model,
    feature_names=feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)


print("DOT representation:\n", dot_data)

